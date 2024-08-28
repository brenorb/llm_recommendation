import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split

RATINGS_PATH = os.path.join(os.pardir, 'data', 'rating.csv')
ANIME_PATH = os.path.join(os.pardir, 'data', 'anime.csv')

def preprocess_data(RATINGS_PATH: str, ANIME_PATH: str) -> tuple:
    """
    Preprocesses the ratings and anime data.

    Args:
        RATINGS_PATH (str): Path to the ratings CSV file.
        ANIME_PATH (str): Path to the anime CSV file.

    Returns:
        tuple: A tuple containing:
            - ratings (DataFrame): The processed ratings DataFrame.
            - anime (DataFrame): The processed anime DataFrame.
    """
    ratings = pd.read_csv(RATINGS_PATH)
    anime = pd.read_csv(ANIME_PATH)

    # Replace "Unknown" in episodes with -1 and convert to int
    anime['episodes'] = anime['episodes'].replace('Unknown', '-1')
    anime['episodes'] = anime['episodes'].astype(int)

    # Fill NaN genres and types with "Unknown"
    anime['genre'] = anime['genre'].fillna('Unknown')
    anime['type'] = anime['type'].fillna('Unknown').astype(str)

    # Fill NaN ratings with -1
    anime['rating'] = anime['rating'].fillna(-1)

    ratings = ratings.drop_duplicates(subset=['user_id', 'anime_id'], keep='last')
    anime = anime.drop_duplicates(subset=['anime_id'], keep='last')

    # all_genres = anime['genre'].dropna().str.split(',')
    # unique_genres = set([genre.strip() for genres in all_genres for genre in genres])
    
    # for genre in sorted(unique_genres):
    #     anime[genre] = anime['genre'].str.contains(genre).astype(int)

    return ratings, anime

def test_split(ratings: pd.DataFrame) -> tuple:
    """
    Splits the ratings data into training and testing sets.

    Args:
        ratings (DataFrame): The ratings DataFrame.

    Returns:
        tuple: A tuple containing:
            - rating_train (DataFrame): The training DataFrame.
            - rating_test (DataFrame): The testing DataFrame.
    """
    user_counts = ratings['user_id'].value_counts()
    frequent_users = user_counts[user_counts > 20].index
    
    # Get frequent and rated subset
    rated = ratings[ratings['rating'] >= 0 & ratings['user_id'].isin(frequent_users)]
    top_user_ratings = rated.groupby('user_id')['rating'].quantile(0.75)
    bottom_user_ratings = rated.groupby('user_id')['rating'].quantile(0.25)
    high_rating_mask = rated['rating'] >= rated['user_id'].map(top_user_ratings)
    low_rating_mask = rated['rating'] <= rated['user_id'].map(bottom_user_ratings)
    x_rated = rated[high_rating_mask | low_rating_mask]

    rating_train, rating_test = train_test_split(x_rated, test_size=0.3, random_state=42)

    return rating_train, rating_test

def feature_engineer(ratings: pd.DataFrame, anime: pd.DataFrame) -> tuple:
    """
    Performs feature engineering on the ratings and anime data.

    Args:
        ratings (DataFrame): The ratings DataFrame.
        anime (DataFrame): The anime DataFrame.

    Returns:
        tuple: A tuple containing:
            - ratings (DataFrame): The processed ratings DataFrame with additional features.
            - features (list): A list of DataFrames containing various feature counts and ratings.
    """
    # Number of anime watched by each user
    ratings['total_watch'] = ratings['user_id'].map(ratings['user_id'].value_counts())
    
    # Average rating for each user
    avg_rating = ratings[ratings['rating'] > 0][['user_id', 'rating']].groupby('user_id').mean().reset_index().rename(columns={'rating': 'avg_rating'})
    ratings = ratings.merge(avg_rating, on='user_id', how='left')

    # Average popularity for each user total anime watched
    user_stats = ratings.merge(anime[['anime_id', 'members']], on='anime_id', how='left')
    user_stats = user_stats.groupby('user_id').mean()['members'].reset_index()
    ratings = ratings.merge(user_stats, on='user_id', how='left').rename(columns={'members': 'avg_members'}, inplace=True)

    # Genre watch counts for each user
    user_genre_data = ratings.merge(anime[['anime_id', 'genre']], on='anime_id', how='left')
    user_genre_data['genre'] = user_genre_data['genre'].fillna('Unknown').str.split(',')
    user_genre_data = user_genre_data.explode('genre')
    user_genre_data['genre'] = user_genre_data['genre'].str.strip()
    genre_counts = pd.pivot_table(user_genre_data, values='rating', index='user_id', 
                                columns='genre', aggfunc='count', fill_value=0)
    # Normalize the counts for each user
    genre_counts = genre_counts.div(genre_counts.sum(axis=1), axis=0)
    genre_counts = genre_counts.reset_index()

    # Type watch counts for each user
    user_type_data = ratings.merge(anime[['anime_id', 'type']], on='anime_id', how='left')
    user_type_data['type'] = user_type_data['type'].fillna('Unknown').astype(str)
    user_type_data = user_type_data.explode('type')
    type_counts = pd.pivot_table(user_type_data, values='rating', index='user_id', 
                                columns='type', aggfunc='count', fill_value=0)
    type_counts = type_counts.div(type_counts.sum(axis=1), axis=0)
    type_counts = type_counts.reset_index()

    # Episode size watch counts for each user
    user_episodes_data = ratings.merge(anime[['anime_id', 'episodes']], on='anime_id', how='left')
    user_episodes_data['episodes'] = user_episodes_data['episodes'].fillna(-1).astype(int)
    user_episodes_data['episodes'] = pd.cut(user_episodes_data['episodes'], bins=[-np.inf, 0, 1, 35, 200, np.inf], labels=['Unknown', 'Movie', 'Short', '6Seasons+', 'Huge'])
    episodes_counts = pd.pivot_table(user_episodes_data, values='rating', index='user_id', 
                                columns='episodes', aggfunc='count', fill_value=0)
    episodes_counts = episodes_counts.div(episodes_counts.sum(axis=1), axis=0)
    episodes_counts = episodes_counts.reset_index()

    # Genre ratings for each user
    g_pos_ratings = user_genre_data[user_genre_data['rating'] > 0]
    g_pos_ratings = g_pos_ratings.explode('genre')
    g_pos_ratings['genre'] = g_pos_ratings['genre'].str.strip()
    genre_ratings = pd.pivot_table(g_pos_ratings, values='rating', index='user_id', 
                                columns='genre', aggfunc='mean', fill_value=0)
    genre_ratings = genre_ratings.reset_index()

    # Type watch ratings for each user
    t_pos_ratings = user_type_data[user_type_data['rating'] > 0]
    t_pos_ratings = t_pos_ratings.explode('type')
    type_ratings = pd.pivot_table(t_pos_ratings, values='rating', index='user_id', 
                                columns='type', aggfunc='mean', fill_value=0)
    type_ratings = type_ratings.reset_index()

    # Episode size watch ratings for each user
    e_pos_ratings = user_episodes_data[user_episodes_data['rating'] > 0]
    e_pos_ratings = e_pos_ratings.explode('episodes')
    episodes_ratings = pd.pivot_table(e_pos_ratings, values='rating', index='user_id', 
                                columns='episodes', aggfunc='mean', fill_value=0)
    episodes_ratings = episodes_ratings.reset_index()

    return ratings, [genre_counts, genre_ratings, episodes_counts, episodes_ratings, type_counts, type_ratings]

def nmf(training_data: pd.DataFrame, redo: bool = False) -> tuple:
    """
    Applies Non-negative Matrix Factorization (NMF) to the training data.

    Args:
        training_data (DataFrame): The training data for NMF.
        redo (bool): Whether to recalculate NMF components.

    Returns:
        tuple: A tuple containing:
            - w1 (ndarray): The user-group matrix from NMF.
            - h1 (ndarray): The group-anime matrix from NMF.
            - user_item_matrix (DataFrame): The user-item matrix.
    """
    if not redo:
        if os.path.exists('data/nmf_components.pkl'):
            print("'nmf_components.pkl' file found.")
            redo = False
        else:
            print("'nmf_components.pkl' file not found. Will need to recalculate NMF components.")
            redo = True

    if training_data.min(axis=1).min() < 0:
        training_data = training_data.add(abs(training_data.min(axis=1).min()), axis=0)

    user_item_matrix = training_data.pivot(index='user_id', columns='anime_id', values='rating')
    
    if redo:
        from sklearn.decomposition import NMF

        dec = NMF(n_components=20, random_state=42)
        w1 = dec.fit_transform(user_item_matrix.fillna(0))  # user-group matrix
        h1 = dec.components_  # group-anime matrix

        # Save w1 and h1 to a pickle file
        with open('data/nmf_components.pkl', 'wb') as f:
            pickle.dump({'w1': w1, 'h1': h1}, f)

        print("NMF components (w1 and h1) have been saved to 'data/nmf_components.pkl'")

    else:
        # Load the NMF components from the pickle file
        with open('data/nmf_components.pkl', 'rb') as f:
            nmf_components = pickle.load(f)
        w1 = nmf_components['w1']
        h1 = nmf_components['h1']
    
    return w1, h1, user_item_matrix

def recommend(user: int, w1: np.ndarray, h1: np.ndarray, user_item_matrix: pd.DataFrame) -> tuple:
    """
    Recommends anime for a specific user based on NMF components.

    Args:
        user (int): The user ID for whom to recommend anime.
        w1 (ndarray): The user-group matrix from NMF.
        h1 (ndarray): The group-anime matrix from NMF.
        user_item_matrix (DataFrame): The user-item matrix.

    Returns:
        tuple: A tuple containing:
            - rec_anime_id (Index): Recommended anime IDs.
            - rec_weight (ndarray): Corresponding weights for the recommendations.
    """
    r_user = user_item_matrix.index.get_loc(user)

    groups = w1.argmax(axis=1)
    user_group = groups[r_user]

    group_users = np.where(groups == user_group)

    group_anime = h1[user_group, :]
    
    # Remove watched anime from recommendations
    watched_anime = user_item_matrix.loc[user].dropna().index
    watched_cols = np.array([user_item_matrix.columns.get_loc(col) for col in watched_anime])

    rec_cols = np.argsort(group_anime)[::-1]
    rec_cols = rec_cols[~np.isin(rec_cols, watched_cols)]

    rec_anime_id = user_item_matrix.iloc[r_user, rec_cols].index
    rec_weight = group_anime[rec_cols] / np.sum(group_anime[rec_cols])
    
    return rec_anime_id, rec_weight

def get_anime_name(user: int, w1: np.ndarray, h1: np.ndarray, user_item_matrix: pd.DataFrame, anime: pd.DataFrame, n: int = 0) -> list:
    """
    Retrieves the names of recommended anime for a user.

    Args:
        user (int): The user ID for whom to retrieve anime names.
        w1 (ndarray): The user-group matrix from NMF.
        h1 (ndarray): The group-anime matrix from NMF.
        user_item_matrix (DataFrame): The user-item matrix.
        anime (DataFrame): The anime DataFrame.
        n (int): The number of recommendations to return.

    Returns:
        list: A list of recommended anime names.
    """
    rec_anime_id, _ = recommend(user, w1, h1, user_item_matrix)

    if n < 0:
        n = 0
    if n > len(rec_anime_id): 
        n = len(rec_anime_id) - 1

    if n:
        return [anime[anime['anime_id'] == anime_id]['name'].values[0] for anime_id in rec_anime_id[:n] if anime_id in anime['anime_id'].values]

    return [anime[anime['anime_id'] == anime_id]['name'].values[0] for anime_id in rec_anime_id if anime_id in anime['anime_id'].values]

if __name__ == '__main__':
    # RATINGS_PATH = '../data/rating.csv'
    # ANIME_PATH = '../data/anime.csv'
    ratings, anime = preprocess_data(RATINGS_PATH, ANIME_PATH)
    rating_train, rating_test = test_split(ratings)
    ratings, features = feature_engineer(ratings, anime)
    w1, h1, user_item_matrix = nmf(rating_train, redo=False)
    rec_anime, rec_anime_names = recommend(1, w1, h1, user_item_matrix, n=5)
    print(rec_anime_names)
    print(rec_anime)
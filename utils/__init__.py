import numpy as np
import pandas as pd
import os
import pickle

RATINGS_PATH = 'data/ratings.csv'
ANIME_PATH = 'data/anime.csv'

def preprocess_data(RATINGS_PATH, ANIME_PATH):
    
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

    all_genres = anime['genre'].dropna().str.split(',')
    unique_genres = set([genre.strip() for genres in all_genres for genre in genres])
    
    for genre in sorted(unique_genres):
        anime[genre] = anime['genre'].str.contains(genre).astype(int)

    return ratings, anime

def test_split(ratings):
    
    # Separate ratings into rated and unrated
    rated = ratings[ratings['rating'] >= 0]
    unrated = ratings[ratings['rating'] < 0]

    # Count the occurrences of each user_id
    user_counts = rated['user_id'].value_counts()

    # Create a mask for users with more than 10 ratings
    frequent_users = user_counts[user_counts > 10].index

    # Create a subset of rated df with users who have more than 10 ratings
    rated_subset = rated[rated['user_id'].isin(frequent_users)]

    # Create a user-anime matrix
    user_anime_matrix = ratings.pivot(index='user_id', columns='anime_id', values='rating')

    # Fill missing values with 0
    user_anime_matrix = user_anime_matrix.fillna(0)

    # Create a mask for train-test split
    mask = np.random.rand(*user_anime_matrix.shape) < 0.8
    train = user_anime_matrix * mask
    test = user_anime_matrix * ~mask

    return train, test

def feature_engineer(ratings, anime):
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

    # Genre ratings for each user
    # Remove unrated rows
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

    return ratings, genre_counts, genre_ratings

def nmf(redo=False):
    # Check if the 'nmf_components.pkl' file exists in the current directory
    if not redo:
        if os.path.exists('nmf_components.pkl'):
            print("'nmf_components.pkl' file found.")
            redo = False
        else:
            print("'nmf_components.pkl' file not found. Will need to recalculate NMF components.")
            redo = True

    user_item_matrix = rated.pivot(index='user_id', columns='anime_id', values='rating')

    if redo:
        from sklearn.decomposition import NMF
        # There are 44 genres, 20 components seems a good place to start
        dec = NMF(n_components=20, random_state=42)
        w1 = dec.fit_transform(user_item_matrix.fillna(0)) # user-group matrix
        h1 = dec.components_ # group-anime matrix

        # Save w1 and h1 to a pickle file
        with open('nmf_components.pkl', 'wb') as f:
            pickle.dump({'w1': w1, 'h1': h1}, f)

        print("NMF components (w1 and h1) have been saved to 'nmf_components.pkl'")

    else:
        # Load the NMF components from the pickle file
        with open('nmf_components.pkl', 'rb') as f:
            nmf_components = pickle.load(f)
        w1 = nmf_components['w1']
        h1 = nmf_components['h1']
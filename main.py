import os
from flask import Flask, request, jsonify
from utils import preprocess_data, nmf, recommend
import pickle 
import pandas as pd
import numpy as np


RATINGS_PATH = './data/rating.csv'
ANIME_PATH = './data/anime.csv'



app = Flask(__name__)

@app.route('/')
def index():
    # def background_task():
    #     # Wait for 30 seconds
    #     time.sleep(30)
    #     print("Background task completed after 30 seconds.")

    # Start the background task
    # thread = threading.Thread(target=background_task)
    # thread.start()

    return 'Welcome to the Anime Recommendation API!'

@app.route('/anime', methods=['GET'])
def get_anime():

    _, anime = preprocess_data(RATINGS_PATH, ANIME_PATH)

    anime_id = request.args.get('anime_id', default=None, type=int)
    name = request.args.get('name', default=None, type=str)
    genre = request.args.get('genre', default=None, type=str)
    type_ = request.args.get('type', default=None, type=str)
    episodes = request.args.get('episodes', default=None, type=int)
    mt_episodes = request.args.get('mt_episodes', default=None, type=int)
    lt_episodes = request.args.get('lt_episodes', default=None, type=int)
    rating = request.args.get('rating', default=None, type=float)
    mt_rating = request.args.get('mt_rating', default=None, type=float)
    lt_rating = request.args.get('lt_rating', default=None, type=float)
    members = request.args.get('members', default=None, type=int)
    mt_members = request.args.get('mt_members', default=None, type=int)
    lt_members = request.args.get('lt_members', default=None, type=int)

    masks = []
    if anime_id:
        masks.append(anime['anime_id'] == anime_id)
    if name:
        masks.append(anime['name'] == name)
    if genre:
        masks.append(anime['genre'].str.contains(genre, case=False))
    if type_:
        masks.append(anime['type'] == type_)
    if episodes:
        masks.append(anime['episodes'] == episodes)
    if mt_episodes:
        masks.append(anime['episodes'] > mt_episodes)
    if lt_episodes:
        masks.append(anime['episodes'] < lt_episodes)
    if rating:
        masks.append(anime['rating'] == rating)
    if mt_rating:
        masks.append(anime['rating'] > mt_rating)
    if lt_rating:
        masks.append(anime['rating'] < lt_rating)
    if members:
        masks.append(anime['members'] == members)
    if mt_members:
        masks.append(anime['members'] > mt_members)
    if lt_members:
        masks.append(anime['members'] < lt_members)

    if masks:
        return anime[np.logical_and.reduce(masks)].to_json(orient='records'), 200
    else:
        return jsonify({'message': 'No data.'}), 200
 
@app.route('/user', methods=['GET'])
def get_user():

    ratings, _ = preprocess_data(RATINGS_PATH, ANIME_PATH)
    
    user_id = request.args.get('user_id', default=None, type=int)
    anime_id = request.args.get('anime_id', default=None, type=int)
    rating = request.args.get('rating', default=None, type=float)
    mt_rating = request.args.get('mt_rating', default=None, type=float)
    lt_rating = request.args.get('lt_rating', default=None, type=float)

    masks = []
    if user_id:
        masks.append(ratings['user_id'] == user_id)
    if anime_id:
        masks.append(ratings['anime_id'] == anime_id)
    if rating:
        masks.append(ratings['rating'] == rating)
    if mt_rating:
        masks.append(ratings['rating'] > mt_rating)
    if lt_rating:
        masks.append(ratings['rating'] < lt_rating)

    if masks:
        return ratings[np.logical_and.reduce(masks)].to_json(orient='records'), 200
    else:
        return jsonify({'message': 'No data.'}), 200


@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend_anime(user_id):
    # Load from pickle file
    with open('model_data.pkl', 'rb') as f:
        ratings, anime, w1, h1, user_item_matrix = pickle.load(f)

    k = request.args.get('k', default=5, type=int)  # Get the top k from query parameters
    rec_anime, rec_weight = recommend(user_id, w1, h1, user_item_matrix)
    rec_anime_names = [anime[anime['anime_id'] == idx]['name'].values[0] for idx in rec_anime[:k]]  # Limit to top k

    return jsonify({'recommended_anime': rec_anime_names, 'weights': rec_weight[:k].tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)
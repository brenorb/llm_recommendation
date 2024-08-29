import os
from flask import Flask, request, jsonify
from utils import preprocess_data, nmf, get_anime_name
from llm import askgpt, response
import pickle 
import pandas as pd
import numpy as np


RATINGS_PATH = './data/rating.csv'
ANIME_PATH = './data/anime.csv'

ratings, anime = preprocess_data(RATINGS_PATH, ANIME_PATH)
training_data = ratings[ratings.rating > 0]

# Uncomment the following line to redo NMF components
# w1, h1, user_item_matrix = nmf(training_data, redo=True)

if os.path.exists('data/nmf_components.pkl'):
    with open('data/nmf_components.pkl', 'rb') as f:
        nmf_components = pickle.load(f)
    w1 = nmf_components['w1']
    h1 = nmf_components['h1']
    user_item_matrix = training_data.pivot(index='user_id', columns='anime_id', values='rating')
else:
    print('NMF components not found. Will need to recalculate NMF components.')
    w1, h1, user_item_matrix = nmf(training_data, redo=True)

app = Flask(__name__)

def recommend_anime(user_id, k=10):
    global w1, h1, user_item_matrix, anime   
    rec_anime_names = get_anime_name(user_id, w1, h1, user_item_matrix, anime, n=k)
    print(rec_anime_names)
    return rec_anime_names

@app.route('/')
def index():

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
def recommend_anime_api(user_id):
    k = request.args.get('k', default=5, type=int)  # Get the top k from query parameters
    # rec_anime_names = recommend_anime(user_id, k)
    rec_anime_names = get_anime_name(user_id, w1, h1, user_item_matrix, anime, n=k)
    return jsonify({'recommended_anime': rec_anime_names}), 200

@app.route('/talk/<int:user_id>/<string:message>', methods=['GET'])
def llm(user_id, message):
    is_debug_mode = app.debug
    sys = '''You are an anime recommendation assistant. You will be provided with prompt request for recommendation and a list of anime. Based on the user ID and the message, you will recommend anime to the user. 
    Pick one anime from the list and recommend it to the user.'''
    rec_anime_names = recommend_anime(user_id)

    message = 'My most likely unwatched animes to like: ' + str(rec_anime_names) + '\n\n' + str(message)
    ans = response(askgpt(message, system=sys))
    global w1, h1, user_item_matrix
    if is_debug_mode:
        return jsonify({'rec': ans, 'sys': sys, 'prompt': message, 'debug': True, 'W1': w1.tolist(), 'H1': h1.tolist(), 'user_item_matrix': user_item_matrix.to_dict('records')}), 200
    else:
        return jsonify({'rec': ans}), 200


if __name__ == '__main__':
    app.run(debug=True)

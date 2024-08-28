import os
from flask import Flask, request, jsonify
from utils import preprocess_data, nmf, recommend
import pickle  # Add this import
import threading  # Add this import
import time
import pandas as pd


RATINGS_PATH = '/data/up_rating.csv'
ANIME_PATH = '/data/up_anime.csv'

app = Flask(__name__)

@app.route('/')
def index():
    def background_task():
        # Wait for 30 seconds
        time.sleep(30)
        print("Background task completed after 30 seconds.")

    # Start the background task
    thread = threading.Thread(target=background_task)
    thread.start()

    return 'Welcome to the Anime Recommendation API!'

@app.route('/upload', methods=['POST'])
def upload_files():
    # if 'rating' not in request.files or 'anime' not in request.files:
    #     print(request.files)
    #     return jsonify({'error': 'Please upload both ratings and anime CSV files.'}), 400

    # ratings_file = request.files['ratings']
    # anime_file = request.files['anime']
    data = request.get_json()
    ratings_file = pd.DataFrame(data['ratings'])
    anime_file = pd.DataFrame(data['anime'])

    ratings_file.to_csv(RATINGS_PATH, index=False)
    anime_file.to_csv(ANIME_PATH, index=False)

    # ratings_file.save(RATINGS_PATH)
    # anime_file.save(ANIME_PATH)

    ratings, anime = preprocess_data(RATINGS_PATH, ANIME_PATH)
    ratings, _ = feature_engineer(ratings, anime)
    # w1, h1, user_item_matrix = nmf(ratings, redo=False)
    w1, h1, user_item_matrix = 1, 2, 3  # mock data

    # Save to pickle file
    with open('model_data.pkl', 'wb') as f:
        pickle.dump((ratings, anime, w1, h1, user_item_matrix), f)

    return jsonify({'message': 'Files uploaded successfully.'}), 200

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
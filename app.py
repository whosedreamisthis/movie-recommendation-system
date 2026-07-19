from flask import Flask, jsonify, request
import joblib
import pandas as pd
from surprise import SVD

app = Flask(__name__)

# Load model and data once at startup
model = joblib.load('models/svd_model.pkl')
movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('userId'))
    # (Insert your logic from predict.py here to get recommendations)
    # Return as JSON
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run()
from flask import Flask, jsonify, request
import joblib
import pandas as pd
from surprise import SVD
from src.predict import get_readable_recommendations

app = Flask(__name__)

# Load model and data once at startup
model = joblib.load('models/svd_model.pkl')
movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')

@app.route('/', methods=['GET'])
def home():
    return {"message": "Welcome to the Movie Recommendation API! Use /recommend?userId=1 to get results."}

@app.route('/recommend', methods=['GET'])
def recommend():
    # Get the userId from arguments
    user_id_param = request.args.get('userId')
    
    # Check if user_id_param exists and is digits (to prevent error)
    if user_id_param is None or not user_id_param.isdigit():
        return jsonify({"error": "Invalid or missing userId"}), 400
    
    user_id = int(user_id_param)
    
    # Now proceed with your existing logic
    recommendations = get_readable_recommendations(user_id)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run()
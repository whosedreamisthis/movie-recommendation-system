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

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"error": "userId is required"}), 400
    
    # Call your prediction logic
    # Note: Make sure get_readable_recommendations is defined 
    # or imported in app.py
    try:
        recommendations = get_readable_recommendations(int(user_id), n=10)
        return jsonify(recommendations.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
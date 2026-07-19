from flask import Flask, jsonify, request
from flask_cors import CORS
from src.predict import get_readable_recommendations

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return {"message": "Welcome to the Movie Recommendation API!"}

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id_param = request.args.get('userId')
    
    if user_id_param is None or not user_id_param.isdigit():
        return jsonify({"error": "Invalid or missing userId"}), 400
    
    user_id = int(user_id_param)
    
    # Capture the result
    recommendations = get_readable_recommendations(user_id)
    
    # Defensive check: if it's None or empty, return a specific error
    if recommendations is None or recommendations.empty:
        return jsonify({"error": "No recommendations found for this user"}), 404
        
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run()
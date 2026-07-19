import joblib
import pandas as pd
import os

# Get the directory where predict.py is located
PREDICT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the PREDICT_DIR
# We move up one level (..) from src/ to access models/ and data/
MODEL_PATH = os.path.join(PREDICT_DIR, '..', 'models', 'svd_model.pkl')
MOVIES_PATH = os.path.join(PREDICT_DIR, '..', 'data', 'movies.csv')
RATINGS_PATH = os.path.join(PREDICT_DIR, '..', 'data', 'ratings.csv')

def get_readable_recommendations(user_id, n=10):
    # 1. Load model and data using robust paths
    model = joblib.load(MODEL_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)
    
    # Get all movie IDs
    all_movie_ids = movies_df['movieId'].unique()
    seen_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    
    # 2. Predict ratings for all unseen movies
    predictions = []
    for m_id in all_movie_ids:
        if m_id not in seen_movies:
            # model.predict returns an object, .est is the estimated rating
            pred = model.predict(uid=user_id, iid=m_id)
            predictions.append((m_id, pred.est))
            
    # 3. Sort by predicted rating and get top N
    pred_df = pd.DataFrame(predictions, columns=['movieId', 'est_rating'])
    top_n_ids = pred_df.sort_values(by='est_rating', ascending=False).head(n)['movieId']
    
    recommendations = movies_df[movies_df['movieId'].isin(top_n_ids)]
    recommendations = recommendations.set_index('movieId').reindex(top_n_ids).reset_index()
    
    return recommendations[['movieId', 'title', 'genres']]

def validate_recommendations(user_id, n=5):
    # 1. Load data using robust paths
    movies_df = pd.read_csv(MOVIES_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)
    
    # 2. Get history
    user_history = ratings_df[ratings_df['userId'] == user_id].sort_values(by='rating', ascending=False)
    top_historical = pd.merge(user_history.head(n), movies_df, on='movieId')
    
    # 3. Get recommendations
    recommendations = get_readable_recommendations(user_id, n=n)
    
    print(f"--- User {user_id} Historical Favorites ---")
    print(top_historical[['title', 'rating']])
    print(f"\n--- New Model Recommendations (Excluding Seen) for User {user_id} ---")
    print(recommendations[['title', 'genres']])

if __name__ == "__main__":
    validate_recommendations(1)
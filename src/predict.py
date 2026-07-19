import joblib
import pandas as pd

def get_recommendations(user_id, n=10):
    try:
        # 1. Load the pre-computed predictions
        predicted_df = joblib.load('../models/predicted_df.pkl')
        
        # 2. Verify user exists
        if user_id not in predicted_df.index:
            return f"User {user_id} not found in the training data."
        
        # 3. Get recommendations
        user_predictions = predicted_df.loc[user_id]
        recommendations = user_predictions.sort_values(ascending=False).head(n)
        
        return recommendations
        
    except FileNotFoundError:
        return "Model artifacts not found. Please run train.py first."

if __name__ == "__main__":
    # Test with user 1
    target_user = 1
    print(f"Top 10 recommendations for user {target_user}:")
    print(get_recommendations(target_user))
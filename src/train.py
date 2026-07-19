import pandas as pd
import joblib
import os
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_and_save_model():
    # 1. Ingestion
    # Ensure paths match your project structure
    ratings = pd.read_csv('../data/ratings.csv')
    
    # 2. Preparation (Surprise expects userId, movieId, rating columns)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Split for evaluation
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # 3. Training
    # SVD in surprise is optimized for recommendation bias
    model = SVD(n_factors=100, random_state=42)
    model.fit(trainset)
    
    # 4. Evaluation
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"Model RMSE: {rmse:.4f}")
    
    # 5. Serialization
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/svd_model.pkl')
    print("Training complete. Surprise SVD model saved to ../models/svd_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
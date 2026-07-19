import pandas as pd
import joblib
import os
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def train_and_save_model():
    # 1. Ingestion
    ratings = pd.read_csv('../data/ratings.csv')
    movies = pd.read_csv('../data/movies.csv')
    df = pd.merge(ratings, movies, on='movieId').dropna()
    
    # 2. Preparation
    user_movie_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    sparse_matrix = csr_matrix(user_movie_matrix.values)
    
    # 3. Training
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix_factorized = svd.fit_transform(sparse_matrix)
    
    # 4. Inference Mapping
    predicted_ratings = svd.inverse_transform(matrix_factorized)
    predicted_df = pd.DataFrame(
        predicted_ratings, 
        columns=user_movie_matrix.columns, 
        index=user_movie_matrix.index
    )
    
    # 5. Serialization
    os.makedirs('models', exist_ok=True)
    joblib.dump(svd, '../models/svd_model.pkl')
    joblib.dump(predicted_df, '../models/predicted_df.pkl')
    print("Training complete. Artifacts saved to models/")

if __name__ == "__main__":
    train_and_save_model()
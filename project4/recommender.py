import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Adjust these paths if you’ve placed your CSVs elsewhere
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'svd.npz')

def train_model(n_components=20):
    """
    1) Load ratings.csv and movies.csv
    2) Build a user×item sparse matrix
    3) Fit TruncatedSVD on that sparse matrix
    4) Save U, Sigma, VT, plus user/movie indices & titles
    """
    # Load raw CSVs
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
    movies  = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))

    # Create pivot table: rows=userId, cols=movieId
    # Use a sparse matrix to save memory
    user_ids  = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_to_idx  = {u:i for i,u in enumerate(user_ids)}
    movie_to_idx = {m:i for i,m in enumerate(movie_ids)}

    rows = ratings['userId'].map(user_to_idx)
    cols = ratings['movieId'].map(movie_to_idx)
    data = ratings['rating'].values

    R_sparse = csr_matrix((data, (rows, cols)),
                          shape=(len(user_ids), len(movie_ids)))

    # Fit SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U     = svd.fit_transform(R_sparse)    # (n_users × k)
    Sigma = svd.singular_values_           # (k,)
    VT    = svd.components_                # (k × n_items)

    # Build movieId→title map
    movie_titles = dict(zip(movies.movieId, movies.title))

    # Persist everything
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    np.savez(
        MODEL_PATH,
        U=U,
        Sigma=Sigma,
        VT=VT,
        user_ids=user_ids,
        movie_ids=movie_ids,
        movie_titles=movie_titles
    )
    return U, Sigma, VT, user_ids, movie_ids, movie_titles

def load_model():
    """
    Load SVD factors + indices + titles from disk.
    """
    data = np.load(MODEL_PATH, allow_pickle=True)
    return (
        data['U'],
        data['Sigma'],
        data['VT'],
        data['user_ids'],
        data['movie_ids'],
        data['movie_titles'].item()   # back to dict
    )

def compute_new_user_profile(ratings_dict, VT, movie_ids, lam=0.1):
    """
    ratings_dict: {movieId: rating}
    Uses closed-form: u = (V_sub^T V_sub + λI)^{-1} V_sub^T r_sub
    """
    # find indices in VT corresponding to rated movieIds
    idxs = [int(np.where(movie_ids == m)[0]) for m in ratings_dict]
    V_sub = VT[:, idxs].T                # (n_rated × k)
    r_sub = np.array([ratings_dict[m] for m in ratings_dict])

    # solve (V_sub^T V_sub + λI) u = V_sub^T r_sub
    A = V_sub.T @ V_sub + lam * np.eye(V_sub.shape[1])
    b = V_sub.T @ r_sub
    u = np.linalg.solve(A, b)            # (k,)
    return u

def recommend(u, VT, movie_ids, movie_titles, top_n=5):
    """
    Given a user vector u (k,), returns top_n movie titles.
    """
    preds = u @ VT                       # (n_items,)
    top_idxs = np.argsort(preds)[-top_n:][::-1]
    rec_ids  = movie_ids[top_idxs]
    return [movie_titles[mid] for mid in rec_ids]

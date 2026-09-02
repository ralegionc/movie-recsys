"""Content features and a content-based recommender for cold-start.

Item features combine:
  - multi-hot genres
  - TF-IDF over the genre/tag text
  - a normalised release-year scalar

These features serve two purposes:
  1. The two-tower item tower consumes them so that *new* items (never seen in
     training) still receive a meaningful embedding -> item cold-start.
  2. A standalone content recommender ranks items for a *new user* given a few
     liked movie ids -> user cold-start fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import DATA_PROC, ARTIFACTS


def build_item_features(movies: pd.DataFrame | None = None):
    """Return (feature_matrix [n_items, d], metadata dict). Row order = item_idx."""
    if movies is None:
        movies = pd.read_parquet(DATA_PROC / "movies.parquet")
    movies = movies.sort_values("item_idx").reset_index(drop=True)

    genres = movies["genres"].fillna("").str.replace("(no genres listed)", "", regex=False)
    genre_lists = [g.split("|") if g else [] for g in genres]

    mlb = MultiLabelBinarizer()
    genre_multi_hot = mlb.fit_transform(genre_lists).astype(np.float32)

    tfidf = TfidfVectorizer(token_pattern=r"[^|]+", max_features=512)
    genre_text = genres.str.replace("|", " ", regex=False)
    tfidf_mat = tfidf.fit_transform(genre_text).astype(np.float32)

    year = movies["year"].to_numpy(dtype=np.float32)
    year = np.nan_to_num(year, nan=np.nanmedian(year))
    year_norm = ((year - year.min()) / (year.max() - year.min() + 1e-9)).reshape(-1, 1)

    feats = sparse.hstack([
        sparse.csr_matrix(genre_multi_hot),
        tfidf_mat,
        sparse.csr_matrix(year_norm),
    ]).tocsr().astype(np.float32)

    meta = {
        "genre_classes": list(mlb.classes_),
        "tfidf_vocab_size": len(tfidf.vocabulary_),
        "n_features": feats.shape[1],
    }
    return feats, meta


class ContentRecommender:
    """Cosine-similarity content recommender for cold-start users."""

    def __init__(self, features: sparse.csr_matrix):
        # L2-normalise rows so dot product == cosine similarity.
        norms = np.sqrt(features.multiply(features).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self.feats = features.multiply(1.0 / norms[:, None]).tocsr()

    def recommend_from_liked(self, liked_item_idx, k: int = 20, exclude=True):
        """Given item indices a new user liked, return top-k similar item indices."""
        liked_item_idx = [i for i in liked_item_idx if 0 <= i < self.feats.shape[0]]
        if not liked_item_idx:
            return []
        profile = self.feats[liked_item_idx].mean(axis=0)
        profile = np.asarray(profile).ravel()
        scores = self.feats.dot(profile)
        if exclude:
            scores[liked_item_idx] = -np.inf
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return top[np.argsort(-scores[top])].tolist()


def save_features():
    feats, meta = build_item_features()
    sparse.save_npz(ARTIFACTS / "item_features.npz", feats)
    print(f"[content] saved item_features.npz shape={feats.shape} meta={meta}")


if __name__ == "__main__":
    save_features()

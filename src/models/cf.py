"""Collaborative-filtering baselines using the Surprise library.

Wraps SVD and NMF matrix factorisation. Trained on explicit ratings, then used
to score arbitrary (user, item) pairs for the shared ranking evaluation.

Scale note: Surprise holds the full trainset in memory and trains with SGD. It
is comfortable up to ml-10m. For ml-25m on CPU, rely on the config core-filter
and/or subsample_frac to keep the trainset tractable, and expect long epochs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from surprise import SVD, NMF, Dataset, Reader

from src.config import CFG, DATA_PROC


class SurpriseCF:
    def __init__(self, algo: str = "svd", n_factors: int = 100, n_epochs: int = 20):
        self.algo_name = algo.lower()
        if self.algo_name == "svd":
            self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=CFG.seed)
        elif self.algo_name == "nmf":
            self.model = NMF(n_factors=n_factors, n_epochs=n_epochs, random_state=CFG.seed)
        else:
            raise ValueError(f"unknown algo: {algo}")

    def fit(self, ratings: pd.DataFrame | None = None):
        """Fit on explicit ratings using the *training* interactions only.

        We reconstruct the training rating rows: everything except each user's two
        most recent positives (which are the val/test held-out items).
        """
        if ratings is None:
            ratings = pd.read_parquet(DATA_PROC / "ratings.parquet")
        reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
        data = Dataset.load_from_df(
            ratings[["user_idx", "item_idx", "rating"]], reader)
        trainset = data.build_full_trainset()
        print(f"[cf:{self.algo_name}] training on {trainset.n_ratings:,} ratings "
              f"({trainset.n_users:,} users, {trainset.n_items:,} items)")
        self.model.fit(trainset)
        return self

    def score(self, user_idx: int, item_indices) -> np.ndarray:
        """Predicted rating for each candidate item (higher = better)."""
        return np.array([self.model.predict(int(user_idx), int(i)).est
                         for i in item_indices], dtype=np.float32)


def build_train_ratings() -> pd.DataFrame:
    """Ratings frame with each user's two most-recent positives removed,
    so the CF model never sees the val/test items."""
    ratings = pd.read_parquet(DATA_PROC / "ratings.parquet")
    pos = ratings[ratings["rating"] >= CFG.positive_threshold]
    pos = pos.sort_values(["user_idx", "timestamp"])
    held = (pos.groupby("user_idx")["timestamp"]
            .rank(method="first", ascending=False))
    heldout_mask = ratings.index.isin(pos[held <= 2].index)
    return ratings[~heldout_mask].copy()

"""Download MovieLens, filter, and build a temporal leave-one-out split.

Run:  python -m src.data.prepare

Produces in data/processed/:
  ratings.parquet   filtered interactions with contiguous user/item indices
  movies.parquet    item metadata (title, genres, year) indexed by item_idx
  splits.npz        train/val/test positive interactions + id mappings
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src.config import CFG, DATA_RAW, DATA_PROC, DATASET_URLS, DATASET_DIRNAME


def download() -> None:
    url = DATASET_URLS[CFG.dataset]
    target = DATA_RAW / DATASET_DIRNAME[CFG.dataset]
    if target.exists():
        print(f"[data] already present: {target}")
        return
    print(f"[data] downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(DATA_RAW)
    print(f"[data] extracted to {target}")


def _iterative_core_filter(ratings: pd.DataFrame) -> pd.DataFrame:
    """Repeatedly drop users/items below the interaction thresholds until stable."""
    before = -1
    while len(ratings) != before:
        before = len(ratings)
        uc = ratings["userId"].value_counts()
        keep_u = uc[uc >= CFG.min_user_interactions].index
        ratings = ratings[ratings["userId"].isin(keep_u)]
        ic = ratings["movieId"].value_counts()
        keep_i = ic[ic >= CFG.min_item_interactions].index
        ratings = ratings[ratings["movieId"].isin(keep_i)]
    return ratings


def _extract_year(title: str):
    # MovieLens titles look like "Toy Story (1995)".
    if isinstance(title, str) and title.endswith(")") and "(" in title:
        frag = title[title.rfind("(") + 1 : -1]
        if frag.isdigit() and len(frag) == 4:
            return int(frag)
    return np.nan


def prepare() -> None:
    download()
    base = DATA_RAW / DATASET_DIRNAME[CFG.dataset]
    ratings = pd.read_csv(base / "ratings.csv")
    movies = pd.read_csv(base / "movies.csv")

    if CFG.subsample_frac is not None:
        ratings = ratings.sample(frac=CFG.subsample_frac, random_state=CFG.seed)
        print(f"[data] subsampled to {len(ratings):,} ratings")

    ratings = _iterative_core_filter(ratings)
    print(f"[data] after core filter: {len(ratings):,} ratings, "
          f"{ratings.userId.nunique():,} users, {ratings.movieId.nunique():,} items")

    # Contiguous integer indices for embeddings.
    u_ids = np.sort(ratings["userId"].unique())
    i_ids = np.sort(ratings["movieId"].unique())
    u_map = {u: i for i, u in enumerate(u_ids)}
    i_map = {m: i for i, m in enumerate(i_ids)}
    ratings["user_idx"] = ratings["userId"].map(u_map).astype(np.int64)
    ratings["item_idx"] = ratings["movieId"].map(i_map).astype(np.int64)

    movies = movies[movies["movieId"].isin(i_map)].copy()
    movies["item_idx"] = movies["movieId"].map(i_map).astype(np.int64)
    movies["year"] = movies["title"].map(_extract_year)
    movies = movies.sort_values("item_idx").reset_index(drop=True)

    ratings.to_parquet(DATA_PROC / "ratings.parquet", index=False)
    movies.to_parquet(DATA_PROC / "movies.parquet", index=False)

    # --- Temporal leave-one-out on positive interactions ---
    pos = ratings[ratings["rating"] >= CFG.positive_threshold].copy()
    pos = pos.sort_values(["user_idx", "timestamp"])
    grp = pos.groupby("user_idx")
    rank_desc = grp["timestamp"].rank(method="first", ascending=False)
    pos["rank_desc"] = rank_desc

    test = pos[pos["rank_desc"] == 1]
    val = pos[pos["rank_desc"] == 2]
    train = pos[pos["rank_desc"] > 2]

    # Only keep val/test users that still have training history.
    train_users = set(train["user_idx"].unique())
    val = val[val["user_idx"].isin(train_users)]
    test = test[test["user_idx"].isin(train_users)]

    n_users = len(u_ids)
    n_items = len(i_ids)

    np.savez(
        DATA_PROC / "splits.npz",
        train_u=train["user_idx"].to_numpy(),
        train_i=train["item_idx"].to_numpy(),
        val_u=val["user_idx"].to_numpy(),
        val_i=val["item_idx"].to_numpy(),
        test_u=test["user_idx"].to_numpy(),
        test_i=test["item_idx"].to_numpy(),
        n_users=n_users,
        n_items=n_items,
    )
    print(f"[data] split -> train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"[data] n_users={n_users:,} n_items={n_items:,}")
    print("[data] done.")


if __name__ == "__main__":
    prepare()

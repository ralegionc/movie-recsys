"""Central configuration. Edit here or override via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

for _p in (DATA_RAW, DATA_PROC, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # --- Dataset ---
    # "small"  -> ml-latest-small (~100k ratings), for fast iteration.
    # "full"   -> ml-25m (~25M ratings), for headline numbers. Slow on CPU.
    dataset: str = os.getenv("RECSYS_DATASET", "small")

    # Keep only users and items with at least this many interactions.
    # For "full", this is what makes Surprise / two-tower tractable on CPU.
    min_user_interactions: int = 20
    min_item_interactions: int = 20

    # Treat a rating >= this threshold as a positive (implicit) interaction
    # for ranking evaluation. MovieLens ratings are 0.5..5.0.
    positive_threshold: float = 4.0

    # Optionally subsample the ratings frame (fraction of rows) before filtering.
    # Set < 1.0 to make a 25M run finish in reasonable CPU time. None = use all.
    subsample_frac: float | None = None

    # --- Eval protocol ---
    # Per-user temporal leave-one-out: most recent positive -> test,
    # second most recent -> validation. Score positive vs N sampled negatives.
    n_eval_negatives: int = 100
    k: int = 10
    seed: int = 42
    # "popularity" -> negatives sampled proportional to item popularity (honest;
    # removes the popular-vs-obscure shortcut that inflates popularity-biased
    # models like NMF). "uniform" -> the classic (biased) protocol.
    neg_sampling: str = "popularity"

    # --- Two-tower ---
    embedding_dim: int = 64
    content_hidden: int = 128
    # Ablation switch. True = hybrid (id + content). False = collaborative-only
    # (id embeddings, no metadata). Override per-run: RECSYS_USE_CONTENT=0.
    use_content: bool = os.getenv("RECSYS_USE_CONTENT", "1") != "0"
    batch_size: int = 1024
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    triplet_margin: float = 0.2
    # "triplet" (margin ranking) or "contrastive" (in-batch InfoNCE).
    loss: str = "contrastive"
    infonce_temp: float = 0.1
    negatives_per_pos: int = 1  # used by the triplet loss

    # --- API / cache ---
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_ttl_seconds: int = 3600
    api_top_k: int = 20


CFG = Config()

DATASET_URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "full": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}
DATASET_DIRNAME = {"small": "ml-latest-small", "full": "ml-25m"}

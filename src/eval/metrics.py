"""Ranking metrics for top-K recommendation evaluation.

All functions operate on a list of per-user results. Each result is a tuple
(ranked_item_ids, relevant_item_ids) where ranked_item_ids is an ordered list
(best first) and relevant_item_ids is a set/collection of ground-truth positives.

The metrics are the standard information-retrieval definitions. They are computed
per user and then averaged (macro-average), which is the convention used in the
recommender-systems literature for NDCG@K / Precision@K / Recall@K / MAP.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _dcg(gains: Sequence[float]) -> float:
    gains = np.asarray(gains, dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top = ranked[:k]
    gains = [1.0 if item in relevant else 0.0 for item in top]
    dcg = _dcg(gains)
    # Ideal DCG: as many relevant items as possible packed at the top.
    ideal_hits = min(len(relevant), k)
    idcg = _dcg([1.0] * ideal_hits)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant = set(relevant)
    if k == 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Average Precision at K for a single user."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top = ranked[:k]
    score = 0.0
    hits = 0
    for i, item in enumerate(top):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    # Normalise by the smaller of (relevant count, k), the standard AP@K denominator.
    return score / min(len(relevant), k)


def evaluate(results: Sequence[tuple[Sequence[int], Iterable[int]]], k: int = 10) -> dict[str, float]:
    """Compute macro-averaged ranking metrics over all users.

    results: list of (ranked_item_ids, relevant_item_ids).
    Returns a dict with NDCG@k, Precision@k, Recall@k, MAP@k.
    """
    if not results:
        return {f"NDCG@{k}": 0.0, f"Precision@{k}": 0.0, f"Recall@{k}": 0.0, f"MAP@{k}": 0.0}

    ndcg = np.mean([ndcg_at_k(r, rel, k) for r, rel in results])
    prec = np.mean([precision_at_k(r, rel, k) for r, rel in results])
    rec = np.mean([recall_at_k(r, rel, k) for r, rel in results])
    mapk = np.mean([average_precision_at_k(r, rel, k) for r, rel in results])
    return {
        f"NDCG@{k}": float(ndcg),
        f"Precision@{k}": float(prec),
        f"Recall@{k}": float(rec),
        f"MAP@{k}": float(mapk),
    }

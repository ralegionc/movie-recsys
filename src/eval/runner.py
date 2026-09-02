"""Shared evaluation runner: identical protocol for every model.

Protocol (sampled-negative ranking). For each evaluation user, take their single
held-out positive item and sample N negatives the user has never interacted with.
The model scores these N+1 candidates; we rank them and compute
NDCG@k / Precision@k / Recall@k / MAP@k with the held-out positive as the only
relevant item. Any object exposing `score(user_idx, item_indices) -> np.ndarray`
plugs in, so CF, content, and the two-tower see the same candidates.

Two things make this comparison honest:

1. Negative sampling mode (config.neg_sampling):
   - "popularity": negatives are drawn proportional to item popularity, so a
     negative is on average as popular as the held-out positive. This removes the
     "popular-vs-obscure" shortcut. Under UNIFORM sampling a pure popularity
     ranker (and popularity-dominated models such as Surprise NMF, which has no
     bias terms) score near-perfectly for the wrong reason. Popularity matching
     is the fix. Sampled metrics are still biased vs full-catalogue ranking
     (Krichene & Rendle, 2020), but the relative comparison is now meaningful.
   - "uniform": the classic protocol, kept for reference.

2. Random tie-breaking: candidates are ranked with a random tiebreak so that the
   order in which candidates are listed (the positive is always first) can never
   leak into the ranking when several scores are equal.
"""
from __future__ import annotations

import numpy as np

from src.config import CFG
from src.eval.metrics import evaluate


def build_user_histories(train_u, train_i, n_items):
    seen = {}
    for u, i in zip(train_u, train_i):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


def item_popularity(seen, n_items):
    """Interaction count per item (number of users who interacted with it)."""
    pop = np.zeros(n_items, dtype=np.float64)
    for items in seen.values():
        for i in items:
            if 0 <= i < n_items:
                pop[i] += 1.0
    return pop


def sample_candidates(eval_u, eval_i, seen, n_items, n_neg,
                      neg_sampling=CFG.neg_sampling, item_pop=None, seed=CFG.seed):
    """For each (user, positive) build [positive] + n_neg negatives.

    neg_sampling="popularity" draws negatives with probability proportional to
    item popularity (+1 smoothing); "uniform" draws them uniformly.
    """
    rng = np.random.default_rng(seed)
    if neg_sampling == "popularity":
        if item_pop is None:
            item_pop = item_popularity(seen, n_items)
        prob = item_pop.astype(float) + 1.0
        prob /= prob.sum()
    else:
        prob = None

    per_user = []
    for u, pos in zip(eval_u, eval_i):
        u, pos = int(u), int(pos)
        blocked = seen.get(u, set())
        negs = []
        while len(negs) < n_neg:
            draw = (rng.choice(n_items, size=n_neg * 2, p=prob) if prob is not None
                    else rng.integers(0, n_items, size=n_neg * 2))
            for c in draw:
                c = int(c)
                if c != pos and c not in blocked and c not in negs:
                    negs.append(c)
                    if len(negs) == n_neg:
                        break
        per_user.append((u, pos, [pos] + negs))
    return per_user


def evaluate_model(model, eval_u, eval_i, seen, n_items,
                   n_neg=CFG.n_eval_negatives, k=CFG.k,
                   neg_sampling=CFG.neg_sampling, item_pop=None, seed=CFG.seed):
    """Run the sampled-negative protocol and return the metrics dict."""
    if neg_sampling == "popularity" and item_pop is None:
        item_pop = item_popularity(seen, n_items)
    per_user = sample_candidates(eval_u, eval_i, seen, n_items, n_neg,
                                 neg_sampling, item_pop, seed)
    tie = np.random.default_rng(seed + 1)
    results = []
    for u, pos, candidates in per_user:
        scores = np.asarray(model.score(u, candidates), dtype=float)
        # Primary key: -scores (descending). Secondary: random, to break ties
        # without letting candidate position (positive first) leak in.
        order = np.lexsort((tie.random(len(scores)), -scores))
        ranked = [candidates[j] for j in order]
        results.append((ranked, {pos}))
    return evaluate(results, k=k)

"""Evaluate every model on the identical held-out test protocol and print a
comparison table, including the two-tower's percentage lift over SVD.

Run (after prepare + content + train):  python -m src.benchmark
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse

from src.config import CFG, DATA_PROC, ARTIFACTS
from src.models.cf import SurpriseCF, build_train_ratings
from src.models.content import ContentRecommender
from src.models.two_tower import TwoTower
from src.eval.runner import build_user_histories, evaluate_model
from src.train_two_tower import TwoTowerScorer


class ContentScorer:
    """Content model as a per-pair scorer: cosine(candidate, user's liked profile)."""
    def __init__(self, feats, seen):
        norms = np.sqrt(feats.multiply(feats).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self.f = feats.multiply(1.0 / norms[:, None]).tocsr()
        self.seen = seen

    def score(self, user_idx, item_indices):
        liked = list(self.seen.get(int(user_idx), []))
        if not liked:
            return np.zeros(len(item_indices))
        profile = np.asarray(self.f[liked].mean(axis=0)).ravel()
        return np.asarray(self.f[np.asarray(item_indices)].dot(profile)).ravel()


def main():
    s = np.load(DATA_PROC / "splits.npz")
    n_items = int(s["n_items"])
    seen = build_user_histories(s["train_u"], s["train_i"], n_items)
    eu, ei = s["test_u"], s["test_i"]
    feats = sparse.load_npz(ARTIFACTS / "item_features.npz")

    results = {}

    # Popularity floor: any model that fails to beat this under popularity-matched
    # negatives is not learning genuine preference. This is the diagnostic that
    # exposes popularity-gaming (e.g. an NMF that scores ~0.98 under uniform negs).
    from src.eval.runner import item_popularity
    item_pop = item_popularity(seen, n_items)

    class MostPopular:
        def score(self, user_idx, item_indices):
            return item_pop[np.asarray(item_indices)]

    results["MostPopular"] = evaluate_model(MostPopular(), eu, ei, seen, n_items)

    train_rat = build_train_ratings()
    for name, algo in [("SVD", "svd"), ("NMF", "nmf")]:
        cf = SurpriseCF(algo, n_factors=100, n_epochs=20).fit(train_rat)
        results[name] = evaluate_model(cf, eu, ei, seen, n_items)

    results["Content"] = evaluate_model(ContentScorer(feats, seen), eu, ei, seen, n_items)

    ckpt = torch.load(ARTIFACTS / "two_tower.pt", map_location="cpu", weights_only=False)
    model = TwoTower(ckpt["n_users"], ckpt["n_items"], ckpt["content_dim"],
                     emb_dim=ckpt["emb_dim"], hidden=ckpt["hidden"],
                     use_content=ckpt.get("use_content", True))
    model.load_state_dict(ckpt["state_dict"])
    scorer = TwoTowerScorer(model, torch.tensor(feats.toarray(), dtype=torch.float32))
    results["TwoTower"] = evaluate_model(scorer, eu, ei, seen, n_items)

    metrics = [f"NDCG@{CFG.k}", f"Precision@{CFG.k}", f"Recall@{CFG.k}", f"MAP@{CFG.k}"]
    header = f"{'Model':<10}" + "".join(f"{m:>14}" for m in metrics)
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<10}" + "".join(f"{m[k]:>14.4f}" for k in metrics))

    if "SVD" in results and "TwoTower" in results:
        print("\nTwoTower lift over SVD:")
        for k in metrics:
            base, new = results["SVD"][k], results["TwoTower"][k]
            lift = (new - base) / base * 100 if base > 0 else float("nan")
            print(f"  {k:<14} {lift:+.1f}%")

    # The honest headline: does the model beat the popularity floor?
    if "MostPopular" in results and "TwoTower" in results:
        nd = f"NDCG@{CFG.k}"
        floor, tt = results["MostPopular"][nd], results["TwoTower"][nd]
        verdict = "PASS" if tt > floor else "FAIL (not learning preference)"
        print(f"\nTwoTower vs MostPopular floor ({nd}): "
              f"{tt:.4f} vs {floor:.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()

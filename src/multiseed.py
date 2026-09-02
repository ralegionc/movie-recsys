"""Multi-seed content ablation with a paired significance test.

Run (after data + features):  python -m src.multiseed [n_seeds]

Trains the hybrid and collaborative-only two-towers across several training seeds
and reports mean +/- std for each variant, plus the per-seed content delta
(hybrid - id-only) as a paired comparison. Evaluation candidates are held fixed
across all runs, so the only variation is training stochasticity and hybrid vs
id-only at each seed are scored on identical negatives -- the paired design that
makes a small delta detectable.

The paired t-test answers: is the content improvement distinguishable from
run-to-run noise? With a handful of seeds it is a weak test, so the fraction of
seeds favouring the hybrid is reported alongside it as a robustness check.
"""
from __future__ import annotations

import sys

import numpy as np

from src.config import CFG, DATA_PROC
from src.train_two_tower import train
from src.eval.runner import build_user_histories, item_popularity, evaluate_model

try:
    from scipy import stats
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False


def popularity_floor():
    s = np.load(DATA_PROC / "splits.npz")
    n_items = int(s["n_items"])
    seen = build_user_histories(s["train_u"], s["train_i"], n_items)
    pop = item_popularity(seen, n_items)

    class MostPopular:
        def score(self, u, items):
            return pop[np.asarray(items)]

    return evaluate_model(MostPopular(), s["test_u"], s["test_i"], seen, n_items)[
        f"NDCG@{CFG.k}"]


def summarize(name, values):
    arr = np.asarray(values)
    print(f"  {name:<22} {arr.mean():.4f} +/- {arr.std(ddof=1):.4f}   "
          f"(min {arr.min():.4f}, max {arr.max():.4f})")
    return arr


def main(n_seeds: int = 5):
    nd = f"NDCG@{CFG.k}"
    seeds = list(range(n_seeds))
    hybrid_nd, idonly_nd = [], []
    # Track all four metrics for the delta consistency check.
    metrics = [f"NDCG@{CFG.k}", f"Precision@{CFG.k}", f"Recall@{CFG.k}", f"MAP@{CFG.k}"]
    deltas = {m: [] for m in metrics}

    for seed in seeds:
        print(f"[multiseed] seed {seed} ...", flush=True)
        _, h = train(use_content=True, save=False, verbose=False, seed=seed)
        _, i = train(use_content=False, save=False, verbose=False, seed=seed)
        hybrid_nd.append(h[nd])
        idonly_nd.append(i[nd])
        for m in metrics:
            deltas[m].append(h[m] - i[m])
        print(f"           hybrid {nd}={h[nd]:.4f} | id-only {nd}={i[nd]:.4f} "
              f"| delta={h[nd]-i[nd]:+.4f}")

    floor = popularity_floor()

    print(f"\n=== {n_seeds}-seed ablation ({nd}) ===")
    print(f"  {'MostPopular (floor)':<22} {floor:.4f}")
    h_arr = summarize("TwoTower id-only", idonly_nd)  # order for readability
    h_arr = summarize("TwoTower hybrid", hybrid_nd)

    print(f"\n=== Content contribution (hybrid - id-only), paired over seeds ===")
    for m in metrics:
        d = np.asarray(deltas[m])
        wins = int((d > 0).sum())
        line = (f"  {m:<14} mean {d.mean():+.4f} +/- {d.std(ddof=1):.4f}  "
                f"| favours hybrid in {wins}/{n_seeds} seeds")
        if _HAVE_SCIPY and n_seeds >= 2 and d.std() > 0:
            t, p = stats.ttest_rel(hybrid_nd, idonly_nd) if m == nd else (None, None)
            if m == nd:
                line += f"  | paired t={t:.2f}, p={p:.4f}"
        print(line)

    d_nd = np.asarray(deltas[nd])
    print("\nVerdict:")
    if _HAVE_SCIPY:
        t, p = stats.ttest_rel(hybrid_nd, idonly_nd)
        if p < 0.05 and d_nd.mean() > 0:
            print(f"  Content adds a significant lift on {nd} "
                  f"(mean {d_nd.mean():+.4f}, paired p={p:.4f}).")
        else:
            print(f"  Content lift on {nd} is not significant at p<0.05 "
                  f"(mean {d_nd.mean():+.4f}, paired p={p:.4f}); "
                  f"report it as within run-to-run noise.")
    else:
        print("  scipy not installed; install it for the paired t-test. "
              f"Mean {nd} delta {d_nd.mean():+.4f}, favours hybrid in "
              f"{int((d_nd>0).sum())}/{n_seeds} seeds.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)

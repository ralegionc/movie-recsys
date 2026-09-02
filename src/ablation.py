"""Content ablation: train the two-tower with and without content features and
report the gap, so you can attribute ranking quality to collaborative vs content
signal.

Run (after data + features):  python -m src.ablation

Both variants use identical training config and the identical eval protocol; the
only difference is whether the item tower sees movie metadata. Interpret:
  - id-only clears the popularity floor  -> collaborative signal is real.
  - hybrid > id-only                     -> content adds ranking quality.
  - hybrid >> id-only ~ floor            -> content is doing the heavy lifting.
"""
from __future__ import annotations

import numpy as np

from src.config import CFG
from src.train_two_tower import train
from src.eval.runner import build_user_histories, item_popularity, evaluate_model
from src.config import DATA_PROC


def popularity_floor():
    s = np.load(DATA_PROC / "splits.npz")
    n_items = int(s["n_items"])
    seen = build_user_histories(s["train_u"], s["train_i"], n_items)
    pop = item_popularity(seen, n_items)

    class MostPopular:
        def score(self, u, items):
            return pop[np.asarray(items)]

    return evaluate_model(MostPopular(), s["test_u"], s["test_i"], seen, n_items)


def main():
    print("=== Training hybrid (id + content) ===")
    _, hybrid = train(use_content=True, tag="", verbose=True)
    print("\n=== Training collaborative-only (id) ===")
    _, idonly = train(use_content=False, tag="_idonly", verbose=True)

    floor = popularity_floor()
    metrics = [f"NDCG@{CFG.k}", f"Precision@{CFG.k}", f"Recall@{CFG.k}", f"MAP@{CFG.k}"]

    header = f"{'Variant':<20}" + "".join(f"{m:>14}" for m in metrics)
    print("\n" + header)
    print("-" * len(header))
    for name, m in [("MostPopular (floor)", floor),
                    ("TwoTower id-only", idonly),
                    ("TwoTower hybrid", hybrid)]:
        print(f"{name:<20}" + "".join(f"{m[k]:>14.4f}" for k in metrics))

    nd = f"NDCG@{CFG.k}"
    fl, io, hy = floor[nd], idonly[nd], hybrid[nd]
    print(f"\n{nd} breakdown:")
    print(f"  collaborative signal (id-only - floor) : {io - fl:+.4f}")
    print(f"  content contribution (hybrid - id-only): {hy - io:+.4f}")
    if hy > io > fl:
        print("  -> both signals contribute; the hybrid is justified.")
    elif io <= fl < hy:
        print("  -> content carries the result; id-only alone doesn't beat popularity.")
    elif io > fl and hy <= io:
        print("  -> collaborative signal carries it; content adds little here.")


if __name__ == "__main__":
    main()

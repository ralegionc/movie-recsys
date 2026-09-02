"""Train the two-tower model and evaluate it with the shared ranking protocol.

Run:  python -m src.train_two_tower

Saves to artifacts/:
  two_tower.pt        model weights + config
  item_embeddings.npy full [n_items, emb_dim] matrix (for fast API serving)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse

from src.config import CFG, DATA_PROC, ARTIFACTS
from src.models.two_tower import TwoTower, triplet_loss, infonce_loss
from src.eval.runner import build_user_histories, evaluate_model


class TwoTowerScorer:
    """Adapter exposing .score(user_idx, item_indices) for the eval runner.

    Precomputes all item embeddings once so per-user scoring is a cheap gather.
    """

    def __init__(self, model: TwoTower, item_feats_dense: torch.Tensor):
        model.eval()
        self.model = model
        with torch.no_grad():
            all_idx = torch.arange(item_feats_dense.size(0))
            self.item_vecs = model.embed_item(all_idx, item_feats_dense).cpu().numpy()

    def score(self, user_idx, item_indices):
        with torch.no_grad():
            uv = self.model.embed_user(torch.tensor([int(user_idx)])).cpu().numpy()[0]
        return self.item_vecs[np.asarray(item_indices)] @ uv


def load():
    s = np.load(DATA_PROC / "splits.npz")
    feats = sparse.load_npz(ARTIFACTS / "item_features.npz")
    feats_dense = torch.tensor(feats.toarray(), dtype=torch.float32)
    return s, feats_dense


def train(use_content=None, tag=None, save=True, verbose=True, seed=None):
    """Train the two-tower and return (best_val_ndcg, test_metrics).

    use_content: override CFG.use_content (True=hybrid, False=collaborative-only).
    tag: suffix for saved artifacts so ablation variants don't clobber each other.
         Defaults to "" for hybrid and "_idonly" for the collaborative-only model.
    seed: training seed (model init, batch order, negative sampling). Defaults to
          CFG.seed. Evaluation candidates stay fixed regardless, so runs at
          different training seeds are scored on identical negatives.
    """
    if use_content is None:
        use_content = CFG.use_content
    if tag is None:
        tag = "" if use_content else "_idonly"
    if seed is None:
        seed = CFG.seed

    torch.manual_seed(seed)
    s, feats_dense = load()
    n_users, n_items = int(s["n_users"]), int(s["n_items"])
    content_dim = feats_dense.size(1)

    train_u = torch.tensor(s["train_u"], dtype=torch.long)
    train_i = torch.tensor(s["train_i"], dtype=torch.long)
    seen = build_user_histories(s["train_u"], s["train_i"], n_items)

    model = TwoTower(n_users, n_items, content_dim,
                     emb_dim=CFG.embedding_dim, hidden=CFG.content_hidden,
                     use_content=use_content)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    n = train_u.size(0)
    mode = "hybrid (id+content)" if use_content else "collaborative-only (id)"
    if verbose:
        print(f"[train] {n:,} positive pairs | {n_users:,} users | {n_items:,} items "
              f"| loss={CFG.loss} | mode={mode}")

    best_ndcg, best_state = -1.0, None
    for epoch in range(1, CFG.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, CFG.batch_size):
            batch = perm[start:start + CFG.batch_size]
            u_idx, pos_idx = train_u[batch], train_i[batch]
            u = model.embed_user(u_idx)
            pos = model.embed_item(pos_idx, feats_dense[pos_idx])

            if CFG.loss == "contrastive":
                loss = infonce_loss(u, pos, CFG.infonce_temp)
            else:  # triplet with uniformly sampled negatives
                neg_idx = torch.randint(0, n_items, (u_idx.size(0),))
                neg = model.embed_item(neg_idx, feats_dense[neg_idx])
                loss = triplet_loss(u, pos, neg, CFG.triplet_margin)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * u_idx.size(0)

        scorer = TwoTowerScorer(model, feats_dense)
        m = evaluate_model(scorer, s["val_u"], s["val_i"], seen, n_items)
        ndcg = m[f"NDCG@{CFG.k}"]
        if verbose:
            print(f"[train] epoch {epoch:2d} | loss {total/n:.4f} | "
                  f"val NDCG@{CFG.k} {ndcg:.4f} | val Recall@{CFG.k} {m[f'Recall@{CFG.k}']:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if save:
        torch.save({"state_dict": model.state_dict(),
                    "n_users": n_users, "n_items": n_items,
                    "content_dim": content_dim, "use_content": use_content,
                    "emb_dim": CFG.embedding_dim, "hidden": CFG.content_hidden},
                   ARTIFACTS / f"two_tower{tag}.pt")
        scorer = TwoTowerScorer(model, feats_dense)
        np.save(ARTIFACTS / f"item_embeddings{tag}.npy", scorer.item_vecs)

    scorer = TwoTowerScorer(model, feats_dense)
    test = evaluate_model(scorer, s["test_u"], s["test_i"], seen, n_items)
    if verbose:
        print(f"[train] BEST val NDCG@{CFG.k}={best_ndcg:.4f}")
        print("[train] TEST:", {k: round(v, 4) for k, v in test.items()})
    return best_ndcg, test


if __name__ == "__main__":
    train()

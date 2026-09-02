"""Two-tower retrieval model.

User tower:  embedding(user_id) -> MLP -> L2-normalised user vector.
Item tower:  embedding(item_id) concatenated with a projection of the item's
             content features -> MLP -> L2-normalised item vector.

Because the item tower ingests content features, an item that never appeared in
training still gets a usable embedding from its metadata alone (item cold-start).

Two training objectives are provided:
  - triplet margin loss (anchor=user, positive=item, negative=item)
  - in-batch InfoNCE contrastive loss (each user's positive is contrasted
    against every other item in the batch as an implicit negative)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, content_dim: int,
                 emb_dim: int = 64, hidden: int = 128, use_content: bool = True):
        super().__init__()
        self.use_content = use_content
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.user_tower = Tower(emb_dim, hidden, emb_dim)
        if use_content:
            # Item tower input: id embedding + projected content features.
            self.content_proj = nn.Linear(content_dim, emb_dim)
            self.item_tower = Tower(emb_dim * 2, hidden, emb_dim)
        else:
            # Ablation: collaborative signal only (id embedding, no content).
            # This isolates how much ranking quality comes from the interaction
            # matrix vs. from movie metadata.
            self.content_proj = None
            self.item_tower = Tower(emb_dim, hidden, emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def embed_user(self, user_idx):
        return self.user_tower(self.user_emb(user_idx))

    def embed_item(self, item_idx, content_feats):
        # content_feats: dense tensor [B, content_dim] aligned with item_idx.
        ids = self.item_emb(item_idx)
        if not self.use_content:
            return self.item_tower(ids)
        c = self.content_proj(content_feats)
        return self.item_tower(torch.cat([ids, c], dim=-1))

    def embed_item_coldstart(self, content_feats):
        """Item embedding from content only (unknown id) -> uses mean id embedding.

        Only meaningful with content enabled; an id-only model cannot embed an
        item it has never seen.
        """
        if not self.use_content:
            raise RuntimeError("cold-start requires use_content=True")
        mean_id = self.item_emb.weight.mean(dim=0, keepdim=True).expand(
            content_feats.size(0), -1)
        c = self.content_proj(content_feats)
        return self.item_tower(torch.cat([mean_id, c], dim=-1))


def triplet_loss(u, pos, neg, margin: float = 0.2):
    """Margin ranking on cosine similarity. Vectors are already L2-normalised."""
    pos_sim = (u * pos).sum(dim=-1)
    neg_sim = (u * neg).sum(dim=-1)
    return F.relu(margin - pos_sim + neg_sim).mean()


def infonce_loss(u, pos, temperature: float = 0.1):
    """In-batch contrastive loss.

    Similarity matrix S[i, j] = cos(user_i, item_j). The diagonal holds the true
    (user, positive-item) pairs; every off-diagonal item is an implicit negative.
    """
    logits = u @ pos.t() / temperature          # [B, B]
    labels = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, labels)

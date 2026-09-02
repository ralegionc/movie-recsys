"""FastAPI recommendation service with Redis caching.

Endpoints
  GET  /health
  GET  /recommend/{user_idx}?k=20         warm user, two-tower retrieval
  POST /recommend/coldstart               new user -> content-based fallback

Run:  uvicorn src.api.main:app --reload
Requires: trained artifacts (two_tower.pt, item_embeddings.npy, item_features.npz)
and a running Redis (REDIS_URL). If Redis is unreachable, the service degrades
gracefully and simply skips caching.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import sparse

from src.config import CFG, DATA_PROC, ARTIFACTS
from src.models.two_tower import TwoTower
from src.models.content import ContentRecommender

app = FastAPI(title="Movie Recommender", version="1.0")

STATE: dict = {}


def _connect_redis():
    try:
        import redis
        r = redis.Redis.from_url(CFG.redis_url, decode_responses=True)
        r.ping()
        print("[api] Redis connected")
        return r
    except Exception as e:  # noqa: BLE001
        print(f"[api] Redis unavailable ({e}); caching disabled")
        return None


@app.on_event("startup")
def load_artifacts():
    ckpt = torch.load(ARTIFACTS / "two_tower.pt", map_location="cpu", weights_only=False)
    model = TwoTower(ckpt["n_users"], ckpt["n_items"], ckpt["content_dim"],
                     emb_dim=ckpt["emb_dim"], hidden=ckpt["hidden"],
                     use_content=ckpt.get("use_content", True))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    STATE["model"] = model
    STATE["item_vecs"] = np.load(ARTIFACTS / "item_embeddings.npy")
    STATE["n_items"] = ckpt["n_items"]
    STATE["n_users"] = ckpt["n_users"]

    feats = sparse.load_npz(ARTIFACTS / "item_features.npz")
    STATE["content"] = ContentRecommender(feats)

    movies = pd.read_parquet(DATA_PROC / "movies.parquet").set_index("item_idx")
    STATE["titles"] = movies["title"].to_dict()
    STATE["redis"] = _connect_redis()
    print("[api] artifacts loaded")


def _titles(idxs):
    t = STATE["titles"]
    return [{"item_idx": int(i), "title": t.get(int(i), "?")} for i in idxs]


def _topk_for_user(user_idx: int, k: int):
    model, item_vecs = STATE["model"], STATE["item_vecs"]
    with torch.no_grad():
        uv = model.embed_user(torch.tensor([user_idx])).cpu().numpy()[0]
    scores = item_vecs @ uv
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return top[np.argsort(-scores[top])].tolist()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE,
            "redis": STATE.get("redis") is not None}


@app.get("/recommend/{user_idx}")
def recommend(user_idx: int, k: int = CFG.api_top_k):
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")
    if not (0 <= user_idx < STATE["n_users"]):
        raise HTTPException(404, f"user_idx out of range [0,{STATE['n_users']})")

    r = STATE.get("redis")
    cache_key = f"rec:{user_idx}:{k}"
    if r is not None:
        cached = r.get(cache_key)
        if cached:
            return {"user_idx": user_idx, "cached": True, "items": json.loads(cached)}

    idxs = _topk_for_user(user_idx, k)
    items = _titles(idxs)
    if r is not None:
        r.setex(cache_key, CFG.cache_ttl_seconds, json.dumps(items))
    return {"user_idx": user_idx, "cached": False, "items": items}


class ColdStartRequest(BaseModel):
    liked_item_idx: list[int]
    k: int = CFG.api_top_k


@app.post("/recommend/coldstart")
def coldstart(req: ColdStartRequest):
    if "content" not in STATE:
        raise HTTPException(503, "model not loaded")
    idxs = STATE["content"].recommend_from_liked(req.liked_item_idx, k=req.k)
    return {"cold_start": True, "seed_items": _titles(req.liked_item_idx),
            "items": _titles(idxs)}

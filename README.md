# Hybrid Movie Recommendation Engine

A hybrid recommender on MovieLens that combines collaborative filtering, a
content-based cold-start model, and a two-tower neural retriever, all compared on
one identical ranking protocol and served through a FastAPI + Redis endpoint.

## What's here

| Component | File | Notes |
|---|---|---|
| Data pipeline | `src/data/prepare.py` | download, core-filter, temporal leave-one-out split |
| CF baselines | `src/models/cf.py` | Surprise SVD + NMF |
| Content model | `src/models/content.py` | genre/tag TF-IDF + year; cold-start recommender |
| Two-tower | `src/models/two_tower.py` | id + content towers, triplet **and** in-batch InfoNCE |
| Trainer | `src/train_two_tower.py` | negative sampling, per-epoch ranking eval, best-checkpoint |
| Eval harness | `src/eval/metrics.py`, `src/eval/runner.py` | NDCG@K / P@K / R@K / MAP@K, sampled negatives |
| Benchmark | `src/benchmark.py` | one table, all models, two-tower lift over SVD |
| Serving | `src/api/main.py` | `/recommend/{user}`, `/recommend/coldstart`, Redis cache |

## Quickstart

```bash
pip install -r requirements.txt
make pipeline        # data -> features -> train -> benchmark  (ml-latest-small)
make api             # serve on :8000  (start Redis first, or it degrades gracefully)
```

Example:

```bash
curl localhost:8000/recommend/3?k=10
curl -X POST localhost:8000/recommend/coldstart \
  -H 'content-type: application/json' \
  -d '{"liked_item_idx": [0, 15, 42], "k": 10}'
```

## Design decisions worth knowing

**Evaluation protocol is the load-bearing part.** Per-user temporal
leave-one-out: each user's most recent positive (rating >= 4.0) is the test
item, the second-most-recent is validation. Each held-out positive is ranked
against 100 sampled unseen negatives, and every model scores the *same*
candidate set. That is what makes "two-tower beats SVD by X%" an honest
comparison rather than an artifact of different candidate pools. Sampled-negative
metrics are known to be biased vs full-catalogue ranking (Krichene & Rendle,
2020); flip `n_eval_negatives` up or evaluate against the full catalogue if you
want the unbiased (slower) number for a writeup.

**Cold start.** The two-tower item tower consumes content features, so an unseen
item gets an embedding from its metadata alone. For an unseen *user*, the API
falls back to the content recommender (average the profile of their liked items,
cosine-rank the catalogue).

**Two losses.** `config.loss = "contrastive"` uses in-batch InfoNCE (every other
item in the batch is an implicit negative — efficient, strong default).
`"triplet"` uses margin ranking with uniformly sampled negatives.

## Scale reality (CPU, no GPU)

- **Default is `ml-latest-small`** (~100k ratings) for fast iteration.
- **`RECSYS_DATASET=full`** uses ml-25m. Surprise keeps the whole trainset in
  memory and trains with SGD; it is comfortable to ml-10m but slow at 25m on
  CPU. The config core-filter (`min_user_interactions`, `min_item_interactions`)
  and `subsample_frac` exist to keep the trainset tractable — expect an
  overnight run for full-scale headline numbers, and consider `subsample_frac`
  around 0.2–0.3 for a same-day pass.
- If you ever want a fast full-25m CF number, `implicit` (ALS) or `LightFM`
  (hybrid) are BLAS-backed and finish in minutes; they make a reasonable
  appendix but the named baselines here are the Surprise ones.

## Config

Everything lives in `src/config.py` (or override via env: `RECSYS_DATASET`,
`REDIS_URL`). Key knobs: `embedding_dim`, `loss`, `epochs`, `batch_size`,
`n_eval_negatives`, `positive_threshold`, `min_*_interactions`.

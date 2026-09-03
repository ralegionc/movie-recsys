# Hybrid Movie Recommendation Engine

**Five recommenders, one evaluation protocol, and a popularity baseline that beats
matrix factorisation.**

Recommender comparisons are usually not comparisons. Each model gets evaluated
against its own candidate set, with its own negative sampling, and the resulting
numbers cannot be put in the same table. This project fixes the protocol first and
then compares: collaborative filtering, a content model, and a two-tower neural
retriever, all ranked against the *same* held-out items and the *same* sampled
negatives.

```
$ make pipeline

  dataset    ml-latest-small, core-filtered
             65,888 ratings, 566 users, 1,286 items

  Model            NDCG@10  Precision@10  Recall@10   MAP@10
  ---------------------------------------------------------------
  MostPopular       0.0517       0.0108     0.1080    0.0349
  SVD               0.0469       0.0113     0.1133    0.0274
  NMF               0.0519       0.0106     0.1062    0.0355
  Content           0.0931       0.0198     0.1982    0.0614
  TwoTower          0.2361       0.0455     0.4549    0.1702

  TwoTower lift over SVD
    NDCG@10        +403.3%
    Precision@10   +301.6%
    MAP@10         +521.8%

  TwoTower vs MostPopular floor:  0.2361 vs 0.0517  ->  PASS
```

**Read the top of that table before the bottom.** SVD scores 0.0469 on NDCG@10
and MostPopular scores 0.0517. Textbook matrix factorisation loses to
recommending whatever is popular. NMF ties it. Only the content model and the
two-tower clear the floor meaningfully.

That is why the popularity baseline is in the table at all. It is the check that
stops a recommender project from reporting a real-looking number for a model that
has learned nothing a `ORDER BY count DESC` could not.

---

## Why this is interesting

**The evaluation protocol is the load-bearing part.** Per-user temporal
leave-one-out: each user's most recent positive, meaning a rating of 4.0 or
above, is the test item, and the second-most-recent is validation. Each held-out
positive is ranked against 100 sampled unseen negatives, and every model scores
the identical candidate set. That is what makes "two-tower beats SVD by 403%" an
honest statement rather than an artifact of different candidate pools.

**Temporal, not random, splitting.** Holding out a random interaction lets a model
see a user's later behaviour while predicting their earlier behaviour. Every
production recommender predicts forward in time, so the evaluation should too.

**The bias in sampled negatives is named, not hidden.** Sampled-negative metrics
are known to be biased relative to full-catalogue ranking (Krichene and Rendle,
KDD 2020). `n_eval_negatives` is a config knob: raise it, or evaluate against the
full catalogue, for the unbiased and much slower number.

**Cold start is handled architecturally, not patched on.** The two-tower item
tower consumes content features, so an unseen item gets an embedding from its
metadata alone. For an unseen *user*, the API falls back to the content
recommender, averaging the profile of their liked items and cosine-ranking the
catalogue.

---

## Architecture

```
   MovieLens (ml-latest-small default, ml-25m via RECSYS_DATASET=full)
        |
        v
   src/data/prepare.py      download -> core filter -> temporal LOO split
        |                   min_user_interactions, min_item_interactions
        |                   -> data/processed/{ratings,movies}.parquet, splits.npz
        v
   +--------------------+-------------------------+
   |                    |                         |
   v                    v                         v
 src/models/cf.py   src/models/content.py   src/models/two_tower.py
 Surprise SVD+NMF   genre/tag TF-IDF        id tower + content tower
                    + year, cold-start      triplet OR in-batch InfoNCE
   |                    |                         |
   |                    |                  src/train_two_tower.py
   |                    |                  negative sampling, per-epoch
   |                    |                  ranking eval, best-checkpoint
   +--------------------+-------------------------+
                        |
                        v
   src/eval/metrics.py  NDCG@K, P@K, R@K, MAP@K
   src/eval/runner.py   one candidate set, every model
                        |
                        v
   src/benchmark.py     the table above, plus the MostPopular floor check
                        |
                        v
   src/api/main.py      /recommend/{user}, /recommend/coldstart, Redis cache
```

## Quickstart

```bash
pip install -r requirements.txt
make pipeline        # data -> features -> train -> benchmark
make api             # serve on :8000
```

Start Redis first if you want caching. The API degrades gracefully without it.

```bash
curl localhost:8000/recommend/3?k=10

curl -X POST localhost:8000/recommend/coldstart \
  -H 'content-type: application/json' \
  -d '{"liked_item_idx": [0, 15, 42], "k": 10}'
```

Individual stages: `make data`, `make features`, `make train`, `make bench`.

## Components

| Component | File | Notes |
|---|---|---|
| Data pipeline | `src/data/prepare.py` | Download, core filter, temporal leave-one-out |
| CF baselines | `src/models/cf.py` | Surprise SVD and NMF |
| Content model | `src/models/content.py` | Genre and tag TF-IDF plus year; cold-start capable |
| Two-tower | `src/models/two_tower.py` | Id and content towers, triplet and in-batch InfoNCE |
| Trainer | `src/train_two_tower.py` | Negative sampling, per-epoch ranking eval, best checkpoint |
| Eval harness | `src/eval/metrics.py`, `src/eval/runner.py` | NDCG, P, R, MAP at K on shared negatives |
| Benchmark | `src/benchmark.py` | One table, all models, floor check |
| Ablation | `src/ablation.py` | Id-only versus id-plus-content towers |
| Serving | `src/api/main.py` | FastAPI, Redis cache, cold-start endpoint |

## Two losses

`config.loss = "contrastive"` uses in-batch InfoNCE, where every other item in the
batch is an implicit negative. Efficient, and the stronger default.

`"triplet"` uses margin ranking with uniformly sampled negatives.

## Scale reality on CPU

`ml-latest-small`, roughly 100k ratings, is the default for fast iteration. After
core filtering it is 566 users and 1,286 items, which is small enough that every
number in the table above should be read with that in mind.

`RECSYS_DATASET=full` switches to ml-25m. Surprise holds the whole trainset in
memory and trains with SGD, which is comfortable to ml-10m and slow at 25m on
CPU. `subsample_frac` around 0.2 to 0.3 gives a same-day pass; expect an overnight
run for full-scale headline numbers.

For a fast full-25m CF number, `implicit` (ALS) or `LightFM` are BLAS-backed and
finish in minutes. They make a reasonable appendix, but the named baselines here
are the Surprise ones.

## Config

Everything is in `src/config.py`, with `RECSYS_DATASET` and `REDIS_URL` overridable
by environment. Key knobs: `embedding_dim`, `loss`, `epochs`, `batch_size`,
`n_eval_negatives`, `positive_threshold`, `min_user_interactions`,
`min_item_interactions`, `subsample_frac`.

## Limitations

**566 users is not a benchmark.** The two-tower's four-fold lift is real on this
split and should not be quoted as a general result. Small-catalogue ranking with
100 sampled negatives is an easier problem than production retrieval, and the gap
between neural and classical methods typically narrows as the catalogue grows.

**SVD underperforming MostPopular is partly a data-size artifact.** Matrix
factorisation needs interaction density that 65,888 ratings across 1,286 items
does not provide. The honest reading is that this dataset cannot distinguish good
CF from bad CF, not that CF is bad.

**Sampled negatives inflate all absolute numbers.** Relative ordering within the
table is trustworthy because the candidate set is shared. The absolute NDCG values
are not comparable to papers that rank against the full catalogue.

**Implicit-feedback framing on explicit ratings.** Treating a rating of 4.0 or
above as a positive discards information and imports the biases of the
implicit-feedback setting onto data that did not have them.

## Roadmap

- Full ml-25m run with full-catalogue evaluation, to produce numbers quotable
  without the sampled-negative caveat
- Sequential model, since temporal splitting is already in place and a
  session-aware architecture is the natural next baseline
- Popularity-debiased sampling of evaluation negatives
- Report the id-only versus id-plus-content ablation in the main table, since
  `src/ablation.py` computes it and it is the clearest evidence for the hybrid
  design

## Data

MovieLens, GroupLens Research. `ml-latest-small` by default, `ml-25m` optional.

Krichene and Rendle, "On Sampled Metrics for Item Recommendation", KDD 2020, for
the sampled-negative bias.

## License

Not yet chosen. MIT would be consistent with the rest of these repositories.

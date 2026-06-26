# Strategy 03 — Training corpus size (per-request vs global)

**Hypothesis (the central one).** The production pipeline trains a *fresh* Word2Vec
model **per request**, on the walks of only the original entity + its ~20–50 pool
candidates. That is a tiny corpus for Word2Vec, which needs many co-occurrences to
place vectors well. Training **one model on a large corpus** should help.

**Setup.** N=50, 5 repeats, offline.
- `per_request`: one model per item (production behaviour).
- `global`: one model trained on the walks of *all* evaluation entities.
- `global, train_on=all`: one model trained on *every* entity in the materialized
  graph (~5.3k entities) — the closest local proxy to "proper" large-corpus RDF2Vec.

Random 0.185; embedding reference 0.252.

## Results (N=50, mean ± std)

### At depth 2 (isolating corpus size)
| config | corpus | MRR | Hits@10 |
|---|---|---|---|
| global_full_corpus | all graph entities | 0.245 ± 0.025 | 0.60 |
| global_baseline | all eval entities | 0.228 ± 0.030 | 0.56 |
| global_walks_100 | all eval entities | 0.227 ± 0.023 | 0.61 |
| baseline_per_request | per item (~20–50) | 0.201 ± 0.028 | 0.59 |

### Combined with the depth-1 fix (the real win)
| config | corpus | depth | MRR | Hits@10 |
|---|---|---|---|---|
| **global_depth1** | all eval entities | 1 | **0.432 ± 0.013** | 0.71 |
| **global_depth1_fullcorpus** | all graph entities | 1 | **0.425 ± 0.015** | 0.75 |
| depth_1 | per item | 1 | 0.336 ± 0.030 | 0.68 |

## Findings

1. **The tiny-corpus hypothesis is correct, but secondary on its own.** At depth 2,
   global training beats per-request by +0.03–0.04 MRR (0.201 → 0.23–0.245). Real,
   but modest.
2. **Corpus size and depth are complementary, and together they are decisive.**
   Global training *at depth 1* reaches **0.432** — far above per-request depth-1
   (0.336) and global depth-2 (0.23). Fixing both of the production's mistakes at
   once roughly **doubles** the production MRR (0.201 → 0.432).
3. **Training on the whole graph adds nothing over the eval-set corpus**
   (0.425 vs 0.432). A few thousand entities already form a usable embedding space;
   the bottleneck was never "not enough entities in the universe," it was that the
   per-request corpus of ~20–50 entities is far too small to train Word2Vec.
4. Lower run-to-run variance for global models (std ≈ 0.013–0.025) than per-request
   (std ≈ 0.03), i.e. global training is also **more stable**.

## Takeaway

Train **one** RDF2Vec model on a shared corpus (all candidate entities, or a
precomputed catalogue), **not** a fresh model per request. Combined with depth-1
walks this is the configuration that makes RDF2Vec strongly outperform both the
production config and the embedding reference on the gold metric.

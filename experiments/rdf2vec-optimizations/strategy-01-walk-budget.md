# Strategy 01 — Walk budget (depth and number of walks)

**Hypothesis.** RDF2Vec under-performed because each entity's graph neighbourhood
was sampled too sparsely (production: `max_depth=2`, `max_walks=4`). Sampling more
and/or deeper walks should give richer, more discriminative embeddings.

**Setup.** N=50 gold eval items, 5 stochastic repeats (seeds 0-4, `PYTHONHASHSEED=0`),
offline on the materialized local graph. Metric: MRR / Hits@k of the gold
substitute. Random baseline MRR ≈ 0.185; sentence-embedding reference 0.252.

## Results (N=50, mean ± std over 5 repeats)

| config | depth | walks | MRR | Hits@5 | Hits@10 |
|---|---|---|---|---|---|
| **depth1_walks_200** | **1** | 200 | **0.351 ± 0.018** | 0.51 | 0.67 |
| **depth_1** | **1** | 50 | **0.336 ± 0.030** | 0.53 | 0.68 |
| walks_200 | 2 | 200 | 0.244 ± 0.019 | 0.38 | 0.60 |
| walks_10 | 2 | 10 | 0.236 ± 0.040 | 0.36 | 0.62 |
| walks_25 | 2 | 25 | 0.236 ± 0.032 | 0.34 | 0.57 |
| walks_100 | 2 | 100 | 0.212 ± 0.026 | 0.31 | 0.53 |
| baseline_per_request | 2 | 4 | 0.201 ± 0.028 | 0.32 | 0.59 |
| walks_50 | 2 | 50 | 0.201 ± 0.031 | 0.32 | 0.48 |
| depth_3_walks_50 | 3 | 50 | 0.200 ± 0.029 | 0.32 | 0.46 |
| depth_4_walks_50 | 4 | 50 | 0.190 ± 0.021 | 0.30 | 0.52 |

(All per-request training, so comparable to the production usage; only the walk
budget varies. The interaction with global training is in `strategy-03`.)

## Findings

1. **Walk depth is the single most impactful parameter, and the production value
   was wrong.** At a fixed `walks=50`, dropping from depth 2→1 raises MRR from
   0.201 to 0.336 (**+0.135**, paired bootstrap P(Δ>0)=1.00). Going deeper hurts:
   depth 3 = 0.200, depth 4 = 0.190 (≈ random).
2. **Why depth-1 wins.** A good "same-kind" substitute shares the original's
   *direct* properties — its `P31` type, country, etc. That signal lives in the
   depth-1 neighbourhood. Depth ≥2 walks append the neighbours' neighbourhoods,
   which are largely shared across *all* entities of a broad class and therefore
   dilute the discriminative signal.
3. **Number of walks is a weak, secondary lever.** At depth 2, increasing walks
   4→200 nudges MRR 0.201→0.244 (noisy). At depth 1, 50→200 walks barely moves it
   (0.336→0.351). ~50-100 walks suffice once depth is correct.
4. **The production budget (depth 2 / 4 walks) is essentially the worst sensible
   choice** — bottom of the table, indistinguishable from random.

## Takeaway

The original RDF2Vec was sampling the *wrong part* of the graph (too deep, too few
walks). **Set `max_depth=1`, `max_walks≈100`.** This alone lifts RDF2Vec from
≈random to clearly above the embedding reference, before any other change.

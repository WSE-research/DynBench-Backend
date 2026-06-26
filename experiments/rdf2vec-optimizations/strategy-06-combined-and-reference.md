# Strategy 06 — Combined configuration and comparison to the reference

**Goal.** Combine the lessons of strategies 01–05 into the best RDF2Vec
configuration and compare it head-to-head with the sentence-embedding reference
and the production baseline, on the gold metric.

**Setup.** N=50, 5 repeats, offline. Random 0.185.

## What each strategy taught

| lever | verdict | best setting |
|---|---|---|
| walk depth (01) | **decisive** | `max_depth=1` |
| #walks (01) | weak/secondary | `max_walks≈100` |
| Word2Vec hparams (02) | not the bottleneck | defaults |
| training corpus (03) | **important** | one **global** model |
| walker (04) | not a lever | RandomWalker |
| sampler (05) | not a lever | uniform |

So the optimum is **global training + depth-1 walks** = `global_depth1`.

> Note: the originally-planned `combined_best` config (global + PageRank + depth-3 +
> 200 walks + 200-dim + 100 epochs) was designed *before* the depth result was in.
> Because it uses depth 3, it scores only 0.236 — a cautionary example that piling
> on "more" parameters hurts if the key one (depth) is wrong.

## Final comparison (N=50, mean ± std)

| config | MRR | Hits@5 | Hits@10 | note |
|---|---|---|---|---|
| **global_depth1** | **0.432 ± 0.013** | 0.58 | 0.71 | optimum (global + depth-1) |
| global_depth1_fullcorpus | 0.425 ± 0.015 | 0.58 | 0.75 | + whole-graph corpus (no extra gain) |
| depth_1 (per-request) | 0.336 ± 0.030 | 0.53 | 0.68 | depth-1 alone |
| **embedding_reference** | **0.252 ± 0.000** | 0.36 | 0.54 | sentence-transformer |
| combined_best (depth-3) | 0.236 ± 0.019 | 0.30 | 0.52 | "more is worse" |
| **baseline_per_request** | **0.201 ± 0.028** | 0.32 | 0.59 | production |
| random baseline | 0.185 | ~0.27 | ~0.45 | analytic |

Paired bootstrap (per-item mean reciprocal rank, N=50):
- `global_depth1` vs `baseline_per_request`: ΔMRR ≈ **+0.23**, P(Δ>0)=1.00 — **significant**
- `global_depth1` vs `embedding_reference`: ΔMRR ≈ **+0.18**, P(Δ>0)=1.00 — **significant**
- `depth_1` vs `embedding_reference`: ΔMRR +0.084 [+0.016,+0.154], P=0.99 — **significant**
- `embedding_reference` vs `baseline_per_request`: ΔMRR +0.050, P=0.88 — n.s.

## Findings

1. **Optimised RDF2Vec (`global_depth1`) more than doubles the production config**
   (0.201 → 0.432) and **significantly outperforms the sentence-embedding
   reference** (0.252) on the gold metric.
2. The production config sits at random (0.201 vs 0.185) — the original
   "RDF2Vec is weak" result was a misconfiguration, not an inherent limit.
3. **Crucial caveat (see `CONCLUSION.md`):** the gold metric rewards
   *structural / type / popularity* matching, which is exactly RDF2Vec's depth-1
   signal and exactly how the gold substitutes were generated. This makes the
   comparison favourable to RDF2Vec and does **not** establish that RDF2Vec picks
   more *semantically* appropriate substitutes than the embedding method, which
   optimises a different (textual) notion. The two are complementary; the gold
   metric cannot crown an absolute winner. What it *does* establish, soundly, is
   that **RDF2Vec is not inherently inferior and was badly under-configured.**

## Takeaway

`SUBSTITUTE_METHOD=rdf2vec` should use **one global model with `max_depth=1`,
`max_walks≈100`**, not the production `depth=2 / walks=4 / per-request`. So
configured, RDF2Vec is at least competitive with — and on the structural gold
metric, better than — the sentence-embedding method.

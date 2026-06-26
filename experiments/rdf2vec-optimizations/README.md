# RDF2Vec optimization experiments

**Question.** In the three-way substitute-search comparison, RDF2Vec was the weakest
method (52 % coverage, mean top-similarity 0.44, slowest). Before concluding that
RDF2Vec is unsuitable, we must be sure it is **used correctly** and that **no
reasonable parameterization makes it competitive**. This folder contains a
systematic parameter-optimization study designed to answer that.

This README describes the methodology and the soundness safeguards. Each individual
strategy and its results are written to its own `strategy-*.md` file, and the
overall verdict is in `CONCLUSION.md`.

---

## 1. What "better" means here — the evaluation metric

The original comparison was *behavioural* (coverage, agreement, speed) with **no
ground truth**, so it could not say which method picked *better* substitutes. We fix
that here by exploiting a ground-truth signal already present in the benchmarks:

> Every `DynQALD`/`DynRuBQ` record contains both the original `query` and the
> `new query`. They differ in exactly **one** entity — the **gold substitute** the
> original DynBench process chose for that item.

Diffing the two queries yields a clean `(original entity → gold substitute)` pair
for 1 791 / 1 791 DynQALD and 4 009 / 4 011 DynRuBQ records. For an evaluation
sample we then rank each original entity's candidate set by similarity and record
**where the gold substitute lands**:

- **Hits@k** — gold in the top *k* (k = 1, 3, 5, 10)
- **MRR** — mean reciprocal rank of the gold
- **mean / median rank** of the gold
- **coverage** — fraction of items for which the method produced a usable ranking

## 2. Soundness safeguards

This metric has real limitations; we neutralize each one explicitly rather than
ignore it.

1. **The gold is a single, popularity-matched substitute, not "the" correct one.**
   Across the benchmarks the gold's PageRank is close to the original's (median
   relative gap 13 % for DynQALD, 5 % for DynRuBQ; ~57–66 % within 20 %). So the
   gold ≈ "a same-type entity of similar popularity," and any one item has *many*
   acceptable substitutes. Hits@k against a single gold is therefore a **noisy,
   conservative lower bound**, not an accuracy score.
   → *Mitigation:* we never read RDF2Vec's numbers in absolute terms. Every RDF2Vec
   config is reported next to (a) a **random-ranking baseline** and (b) the
   **sentence-embedding method** (the established best, run on the identical
   candidate sets and data). The question is strictly *relative*: can any RDF2Vec
   config approach the embedding reference?

2. **Retrieval coverage vs ranking quality are different failure modes.** The gold
   is often not even in the recall pool (a property of retrieval, shared by all
   ranking methods). To isolate **ranking** ability we **inject the gold** into the
   candidate set when retrieval missed it, and report the natural-retrieval rate
   separately.

3. **The live-endpoint timeouts confounded the original RDF2Vec coverage.** ~93 % of
   RDF2Vec's empty results in the bulk run were ≥15 s walk-extraction timeouts, not
   ranking failures. To remove this confound we **materialize a bounded local graph
   once** and run every config offline against it — no network, no timeouts, so
   coverage reflects the method, not the endpoint.

4. **Implementation correctness.** A separate sanity check confirms the exact
   embedding code separates clean clusters (Nordic countries vs Pacific islands in
   `tests/data/small_graph.ttl`: Finland~Sweden ≈ 0.996 ≫ Finland~EasterIsland ≈
   0.29). So the embedder itself is correct; the study isolates *parameterization*
   and *usage*, not bugs.

5. **Determinism.** Fixed seed (42), `workers=1`, subsampling off (`sample=0`),
   CRC32 hash — identical to the production embedder — so results are reproducible.

## 3. Data and harness

- `_prepare_eval_data.py` → `data/eval_set.json`: a sample of distinct original
  entities (`EVAL_N`, default 50; `EVAL_POOL_LIMIT` 50; seed 42) with their gold,
  candidate set (label/description text), and natural-retrieval flag.
- `_materialize_graph.py` → `data/local_graph.nt`: the Wikidata *truthy* entity
  sub-graph (entity-valued `wdt:` edges) for all evaluation nodes (depth 1) plus
  their shared neighbours (depth 2, bounded), queried once.
- `run_experiment.py` → `results/<config>.json`: runs each configuration offline
  and writes per-item ranks and an aggregate summary.

## 4. Optimization strategies (each has its own report)

| Report | What it varies | Hypothesis tested |
|---|---|---|
| `strategy-01-walk-budget.md` | walks per entity (4→200), depth (1→4) | richer sampling of each entity's neighbourhood helps |
| `strategy-02-word2vec-hparams.md` | vector size, epochs, window, skip-gram vs CBOW | the Word2Vec training, not the walks, is the bottleneck |
| `strategy-03-corpus-size.md` | per-request training vs **one global model** | **the tiny per-request corpus is the core flaw** |
| `strategy-04-walker.md` | Random / Weisfeiler-Lehman / HALK / NGram / Walklet | a structure-aware walker captures "same kind" better |
| `strategy-05-sampler.md` | uniform / PageRank / object-frequency / wide | biasing walks toward informative edges helps |
| `strategy-06-combined-and-reference.md` | best-of-everything vs embedding reference | the ceiling of optimized RDF2Vec vs the established best |

`CONCLUSION.md` aggregates all of the above into the final verdict on whether
RDF2Vec is correctly used and whether it can be made competitive.

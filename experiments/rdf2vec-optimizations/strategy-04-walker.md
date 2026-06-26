# Strategy 04 — Walking strategy

**Hypothesis.** The default RandomWalker samples generic paths. A structure-aware
walker — Weisfeiler-Lehman (subtree patterns), HALK, NGram, or Walklet — might
capture "same kind" similarity better.

**Setup.** N=50, 5 repeats, offline, global training at `depth=2, walks=100`
(so all rows are directly comparable to `global_walks_100`). Random 0.185;
embedding reference 0.252.

## Results (N=50, mean ± std)

| walker | MRR | Hits@5 | Hits@10 |
|---|---|---|---|
| walklet | 0.235 ± 0.033 | 0.38 | 0.60 |
| ngram | 0.225 ± 0.022 | 0.36 | 0.61 |
| random (= global_walks_100) | 0.227 ± 0.023 | 0.34 | 0.61 |
| weisfeiler_lehman | 0.221 ± 0.028 | 0.33 | 0.61 |
| halk | 0.209 ± 0.031 | 0.32 | 0.52 |

## Findings

1. **Walker choice is not a meaningful lever here.** All five fall in 0.209–0.235,
   overlapping within noise; none beats plain RandomWalker by a significant margin.
2. Weisfeiler-Lehman — the usual "more structural" recommendation — does **not**
   help (0.221), because at depth 2 the discriminative signal is already swamped
   (see `strategy-01`); a fancier walker over the wrong-depth neighbourhood cannot
   fix that.
3. The clean-graph sanity check (`small_graph.ttl`) showed WL gives sharper cluster
   separation than Random; that advantage does not transfer to the noisy, broad
   Wikidata neighbourhoods at this task.

## Takeaway

Keep the default **RandomWalker**. Walker sophistication is not where RDF2Vec's
performance is decided for substitute search; depth and corpus are.

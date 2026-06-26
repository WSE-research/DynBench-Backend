# Strategy 05 — Walk sampler (edge selection bias)

**Hypothesis.** Uniform walk sampling treats all edges equally. Biasing the walk
toward informative edges — by PageRank, object frequency, or the "wide" heuristic —
might produce more discriminative walks.

**Setup.** N=50, 5 repeats, offline, global training at `depth=2, walks=100`.
Baseline sampler is uniform (= `global_walks_100`, MRR 0.227). Random 0.185;
embedding reference 0.252.

## Results (N=50, mean ± std)

| sampler | MRR | Hits@5 | Hits@10 |
|---|---|---|---|
| wide | 0.241 ± 0.015 | 0.33 | 0.59 |
| pagerank | 0.233 ± 0.054 | 0.33 | 0.54 |
| uniform (baseline) | 0.227 ± 0.023 | 0.34 | 0.61 |
| objfreq | 0.196 ± 0.021 | 0.26 | 0.52 |

## Findings

1. **Sampler choice is a minor lever.** Differences (0.196–0.241) are small and
   mostly within noise.
2. The "wide" sampler is marginally best; object-frequency sampling is marginally
   *worse* than uniform (it over-weights popular hub objects, which are shared by
   everything and thus non-discriminative).
3. PageRank sampling has the **highest run-to-run variance** (std 0.054) — an extra
   stability cost for no mean benefit.
4. As with the walker, all of this is at depth 2 and therefore capped well below the
   depth-1 result; sampler tuning cannot compensate for the wrong depth.

## Takeaway

Keep the default **uniform** sampler (or "wide" for a marginal, noisy gain). Sampler
bias is not where the performance is won.

# Strategy 02 — Word2Vec hyperparameters

**Hypothesis.** Maybe the walks are fine and the bottleneck is the Word2Vec
training: too few dimensions / epochs, wrong context window, or skip-gram vs CBOW.

**Setup.** N=50, 5 repeats, offline. Base is a per-request model at `depth=2,
walks=100` (MRR ≈ 0.212); each row changes one Word2Vec hyperparameter. Random
0.185; embedding reference 0.252.

## Results (N=50, mean ± std)

| config | change | MRR | Hits@5 | Hits@10 |
|---|---|---|---|---|
| cbow | `sg=0` (CBOW) | 0.225 ± 0.025 | 0.32 | 0.54 |
| epochs_50 | epochs 10→50 | 0.224 ± 0.007 | 0.34 | 0.54 |
| epochs_100 | epochs 10→100 | 0.224 ± 0.028 | 0.36 | 0.56 |
| window_10 | window 5→10 | 0.217 ± 0.026 | 0.36 | 0.55 |
| walks_100 (base) | — | 0.212 ± 0.026 | 0.31 | 0.53 |
| vec_200 | dim 100→200 | 0.211 ± 0.026 | 0.30 | 0.53 |
| vec_300 | dim 100→300 | 0.210 ± 0.025 | 0.32 | 0.52 |

## Findings

1. **Word2Vec hyperparameters are not the bottleneck.** Every variant lands in a
   tight 0.21–0.225 band — within one standard deviation of the base. None
   approaches depth-1's 0.336 (`strategy-01`).
2. More epochs and CBOW give marginal, non-significant gains; larger vectors give
   nothing (the per-request corpus is too small to fill 200–300 dimensions).
3. This is the expected outcome: if the *input* walks don't carry the
   discriminative signal (because depth is wrong), no amount of Word2Vec tuning
   recovers it.

## Takeaway

Do not spend effort tuning Word2Vec. Keep defaults (`vector_size=100`,
`epochs≈10–50`, `sg=1`, `window=5`). The leverage is in the walks (depth) and the
training corpus (`strategy-03`), not the embedder hyperparameters.

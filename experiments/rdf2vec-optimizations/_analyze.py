"""
Aggregate the RDF2Vec sweep results (with stochastic repeats) into a statistically
grounded leaderboard:

  - MRR mean +/- std ACROSS REPEATS (run-to-run stability of each config)
  - 95% bootstrap CI of MRR OVER ITEMS (using each item's mean reciprocal rank
    across repeats), so sampling uncertainty is explicit
  - Hits@5 / Hits@10 (mean across repeats), random baseline
  - paired bootstrap comparisons for the decisive config pairs

Usage:
    .venv/bin/python experiments/rdf2vec-optimizations/_analyze.py [results_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results")
B = 10000
RNG = np.random.default_rng(42)


def load():
    out = {}
    for f in sorted(RESULTS.glob("*.json")):
        d = json.load(open(f))
        out[d["summary"]["config"]] = d
    return out


def boot_ci(x):
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    s = np.mean(x[idx], axis=1)
    return float(np.mean(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired(a_map, b_map):
    ids = sorted(set(a_map) & set(b_map))
    d = np.array([a_map[i] - b_map[i] for i in ids])
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    s = np.mean(d[idx], axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi), float(np.mean(s > 0))


def main():
    data = load()
    if not data:
        print(f"no results in {RESULTS}")
        return
    any_sum = next(iter(data.values()))["summary"]
    print(f"# {RESULTS.name}: N={any_sum['items']} items, "
          f"repeats={any_sum['repeats']}, bootstrap B={B}\n")

    mrr_item = {name: {p["id"]: p["mean_rr"] for p in d["per_item_mean"]}
                for name, d in data.items()}

    rows = []
    for name, d in data.items():
        s = d["summary"]
        x = np.array([p["mean_rr"] for p in d["per_item_mean"]])
        _, lo, hi = boot_ci(x)
        rows.append((name, s["mode"], s["MRR_mean"], s["MRR_std"], lo, hi,
                     s["Hits@5"], s["Hits@10"], s["random_MRR"], s["coverage"]))
    rows.sort(key=lambda r: r[2], reverse=True)

    print("| config | mode | MRR mean±std | MRR 95% CI (items) | Hits@5 | Hits@10 | randMRR | cov |")
    print("|---|---|---|---|---|---|---|---|")
    for name, mode, mm, sd, lo, hi, h5, h10, rnd, cov in rows:
        print(f"| {name} | {mode} | {mm:.3f}±{sd:.3f} | [{lo:.3f},{hi:.3f}] | "
              f"{h5:.2f} | {h10:.2f} | {rnd:.3f} | {cov:.2f} |")

    print("\n## Paired bootstrap comparisons (per-item mean reciprocal rank)\n")
    best = max((r for r in rows if r[1] != "embedding"), key=lambda r: r[2])[0]
    def cmp(a, b):
        if a in mrr_item and b in mrr_item:
            m, lo, hi, p = paired(mrr_item[a], mrr_item[b])
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
            print(f"- **{a}** vs **{b}**: ΔMRR={m:+.3f} [{lo:+.3f},{hi:+.3f}] P(Δ>0)={p:.2f} → {sig}")
    print(f"(best non-embedding config: {best})")
    cmp(best, "baseline_per_request")
    cmp(best, "embedding_reference")
    cmp("embedding_reference", "baseline_per_request")
    cmp("depth_1", "baseline_per_request")


if __name__ == "__main__":
    main()

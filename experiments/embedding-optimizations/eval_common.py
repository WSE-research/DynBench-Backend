"""
Shared evaluation metric for the substitute-ranking experiments.

These functions are copied VERBATIM from
`experiments/rdf2vec-optimizations/run_experiment.py` so that the embedding
optimization study scores candidates with byte-identical logic to the RDF2Vec
study — that is the whole point: the two studies' results must be directly
comparable. The baseline embedding config in this folder reproduces the
`embedding_reference` already present in the RDF2Vec results, which is the
empirical proof that the harness is comparable. Keep in sync if either changes.

Pure, method-agnostic helpers (no RDF2Vec / no embedding specifics):
  gold_rank_from_sims, metrics_one, random_baseline, aggregate, load_eval.
"""
import json
from pathlib import Path

import numpy as np


def gold_rank_from_sims(item, sim_of):
    """Given sim_of: uri->similarity, return (gold_rank, n_ranked) among candidates with a sim."""
    scored = [(c["uri"], sim_of.get(c["uri"])) for c in item["candidates"]]
    scored = [(u, s) for u, s in scored if s is not None]
    if not scored:
        return None, 0
    scored.sort(key=lambda x: x[1], reverse=True)
    order = [u for u, _ in scored]
    gold = item["gold"]
    rank = order.index(gold) + 1 if gold in order else None
    return rank, len(order)


def metrics_one(per_item):
    """Metrics for a single repeat's per-item ranks."""
    n = len(per_item)
    found = [p["gold_rank"] for p in per_item if p["gold_rank"]]

    def hits(k):
        return sum(1 for r in found if r <= k) / n

    return {"MRR": sum(1.0 / r for r in found) / n,
            "Hits@1": hits(1), "Hits@3": hits(3), "Hits@5": hits(5), "Hits@10": hits(10),
            "coverage": len(found) / n}


def random_baseline(per_item):
    n = len(per_item)
    rmrr = rh5 = 0.0
    for p in per_item:
        m = p["n_ranked"] or p["n_candidates"]
        if m <= 0:
            continue
        rmrr += (sum(1.0 / r for r in range(1, m + 1)) / m) / n
        rh5 += min(5, m) / m / n
    return rmrr, rh5


def aggregate(results_dir, name, mode, cfg, reps, seconds):
    """Aggregate a config's per-repeat per-item ranks into the summary + result file.
    Identical summary schema to the RDF2Vec study so the same `_analyze.py` works and
    the two studies' result files are directly comparable. `seconds` is total compute."""
    import statistics as stx
    nrep = len(reps)
    met = [metrics_one(pi) for pi in reps]

    def mean_std(key):
        vals = [m[key] for m in met]
        return round(stx.mean(vals), 3), (round(stx.pstdev(vals), 3) if nrep > 1 else 0.0)

    mrr_mean, mrr_std = mean_std("MRR")
    rmrr, rh5 = random_baseline(reps[0])

    ids = [p["id"] for p in reps[0]]
    per_item_mean = []
    for idx, iid in enumerate(ids):
        rrs = [(1.0 / pi[idx]["gold_rank"] if pi[idx]["gold_rank"] else 0.0) for pi in reps]
        per_item_mean.append({"id": iid, "mean_rr": sum(rrs) / len(rrs),
                              "n_candidates": reps[0][idx]["n_candidates"],
                              "found_frac": sum(1 for pi in reps if pi[idx]["gold_rank"]) / len(reps)})

    summary = {
        "config": name, "mode": mode, "repeats": nrep, "items": len(reps[0]),
        "MRR_mean": mrr_mean, "MRR_std": mrr_std,
        "MRR_min": round(min(m["MRR"] for m in met), 3),
        "MRR_max": round(max(m["MRR"] for m in met), 3),
        "Hits@1": mean_std("Hits@1")[0], "Hits@3": mean_std("Hits@3")[0],
        "Hits@5": mean_std("Hits@5")[0], "Hits@10": mean_std("Hits@10")[0],
        "coverage": mean_std("coverage")[0],
        "random_MRR": round(rmrr, 3), "random_Hits@5": round(rh5, 3),
        "mean_candidates": round(float(np.mean([p["n_candidates"] for p in reps[0]])), 1),
        "seconds": round(seconds, 1),
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "config": cfg, "per_repeat_metrics": met,
               "per_item_mean": per_item_mean, "per_repeat": reps},
              open(Path(results_dir) / f"{name}.json", "w"), indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def load_eval(eval_file):
    return json.load(open(eval_file, encoding="utf-8"))

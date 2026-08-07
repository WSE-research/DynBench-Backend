"""
Per-language breakdown of a substitute-ranking result file.

Joins a result JSON's per-item ranks (written by either run_experiment.py in this
folder or in ../embedding-optimizations/, both via eval_common.aggregate) back to
the "language" field of the eval_set.json they were computed against, and reports
MRR / Hits@k / coverage separately for each language.

Joined by LIST POSITION, not by the "id" field: benchmark ids (e.g. "train:31")
are only unique WITHIN a language -- DynQALD/DynRuBQ reuse the same id across a
question's per-language translations, so different languages' items can share an
id. run_experiment.py iterates `for it in EVAL` and appends per-item results in
that exact order, so position is the reliable join key.

Usage:
    .venv/bin/python experiments/rdf2vec-optimizations/_analyze_by_language.py \
        data_lang100/eval_set.json <result.json> [<result.json> ...] [--json out.json]
"""
import json
import sys
from pathlib import Path

LANGUAGES = ["en", "de", "fr", "ru", "uk", "udm"]


def metrics_for(per_item):
    n = len(per_item)
    if n == 0:
        return None
    found = [p["gold_rank"] for p in per_item if p["gold_rank"]]

    def hits(k):
        return sum(1 for r in found if r <= k) / n

    return {
        "n": n,
        "MRR": sum(1.0 / r for r in found) / n if found else 0.0,
        "Hits@1": hits(1), "Hits@5": hits(5), "Hits@10": hits(10),
        "coverage": len(found) / n,
    }


def main():
    args = sys.argv[1:]
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = Path(args[i + 1])
        del args[i:i + 2]

    eval_path = Path(args[0])
    result_paths = [Path(p) for p in args[1:]]

    eval_items = json.load(open(eval_path, encoding="utf-8"))
    languages_by_pos = [it["language"] for it in eval_items]

    summary = {}
    for rp in result_paths:
        d = json.load(open(rp, encoding="utf-8"))
        config = d["summary"]["config"]
        # average gold_rank across repeats per item (matches eval_common's per_item_mean logic,
        # but we need per-item gold_rank booleans per language, so recompute from per_repeat)
        reps = d["per_repeat"]
        assert len(reps[0]) == len(languages_by_pos), \
            f"{rp}: {len(reps[0])} result items vs {len(languages_by_pos)} eval items"
        per_item = []
        for idx, lang in enumerate(languages_by_pos):
            ranks = [pi[idx]["gold_rank"] for pi in reps]
            found_ranks = [r for r in ranks if r]
            # "found" if the majority of repeats found it; rank = mean of found ranks (harmonic-ish
            # via mean reciprocal rank is more correct, so store mean_rr instead of a rank)
            mean_rr = sum(1.0 / r for r in found_ranks) / len(reps) if found_ranks else 0.0
            per_item.append({"id": reps[0][idx]["id"], "language": lang, "mean_rr": mean_rr,
                             "found_frac": len(found_ranks) / len(reps)})

        print(f"\n=== {config} ({rp}) ===")
        overall_n = len(per_item)
        overall_mrr = sum(p["mean_rr"] for p in per_item) / overall_n
        overall_cov = sum(p["found_frac"] for p in per_item) / overall_n
        print(f"{'ALL':4s} n={overall_n:4d}  MRR={overall_mrr:.3f}  coverage={overall_cov:.1%}")
        config_summary = {"ALL": {"n": overall_n, "MRR": round(overall_mrr, 3),
                                  "coverage": round(overall_cov, 3)}}
        for lg in LANGUAGES:
            items = [p for p in per_item if p["language"] == lg]
            if not items:
                continue
            n = len(items)
            mrr = sum(p["mean_rr"] for p in items) / n
            cov = sum(p["found_frac"] for p in items) / n
            print(f"{lg:4s} n={n:4d}  MRR={mrr:.3f}  coverage={cov:.1%}")
            config_summary[lg] = {"n": n, "MRR": round(mrr, 3), "coverage": round(cov, 3)}
        summary[config] = {"result_file": str(rp), "mode": d["summary"]["mode"],
                           "repeats": d["summary"]["repeats"], "by_language": config_summary}

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"eval_set": str(eval_path), "languages": LANGUAGES, "configs": summary},
                  open(json_out, "w", encoding="utf-8"), indent=2)
        print(f"\nWrote {json_out}")


if __name__ == "__main__":
    main()

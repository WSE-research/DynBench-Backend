"""
Prepare the gold-based evaluation set for the RDF2Vec optimization experiments.

For each sampled benchmark record we know the gold substitution the original
DynBench process chose (the entity that differs between `query` and `new query`).
We rebuild the shared candidate pool (the same retrieval the embedding/rdf2vec
methods use) and record, for every original entity:

  - the gold substitute and whether it was naturally retrieved into the pool,
  - the candidate set (URIs + label/description text) to be ranked, with the
    gold injected if retrieval missed it (so ranking quality can be measured
    independently of retrieval coverage),
  - label/description text for the original and the gold.

Output: experiments/rdf2vec-optimizations/data/eval_set.json and node_uris.json
(the union of all entity URIs, used to materialize the local graph next).

Run from the DynBench-Backend root with its venv:
    .venv/bin/python experiments/rdf2vec-optimizations/_prepare_eval_data.py
"""
import json
import os
import random
import re
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # DynBench-Backend root

from utils.sparql import execute as raw_execute, normal_sparql
from utils.wikidata import (
    collect_candidate_pool, get_entity_profile, WIKIDATA_PREFIX, short2uri,
)
from tests.test_method_comparison import build_info, LIVE_ENDPOINT, LIVE_AGENT, make_caching_execute

N = int(os.environ.get("EVAL_N", "40"))
POOL_LIMIT = int(os.environ.get("EVAL_POOL_LIMIT", "50"))
SEED = int(os.environ.get("EVAL_SEED", "42"))

OUT = Path(__file__).parent / os.environ.get("EXP_DATA_DIR", "data")
OUT.mkdir(parents=True, exist_ok=True)

ENT = re.compile(r"wd:Q\d+")


def gold_pairs():
    """Yield (record_id, language, query, original_entity, gold_entity) for clean single-entity swaps."""
    root = Path(__file__).resolve().parents[2] / "benchmarks"
    for name in ("DynQALD.json", "DynRuBQ.json"):
        for r in json.load(open(root / name, encoding="utf-8")):
            o, ne = r.get("query", ""), r.get("new query", "")
            so, sne = set(ENT.findall(o)), set(ENT.findall(ne))
            do, dn = so - sne, sne - so
            if len(do) == 1 and len(dn) == 1:
                yield r.get("id"), r.get("language"), o, next(iter(do)), next(iter(dn))


def main():
    raw = partial(raw_execute, endpoint_url=LIVE_ENDPOINT, agent=LIVE_AGENT, delay=0.4, timeout=30.0)
    execute = make_caching_execute(raw, {})

    pairs = list(gold_pairs())
    random.Random(SEED).shuffle(pairs)

    seen_original = set()
    eval_items = []
    node_uris = set()

    for rid, lang, query, original, gold in pairs:
        if len(eval_items) >= N:
            break
        if original in seen_original:
            continue
        try:
            nquery = normal_sparql(query, WIKIDATA_PREFIX)
            info = build_info(nquery, original, execute)
            if not info.get("conditions", {}).get(original):
                continue  # no type/query conditions -> no pool can be built
            pool, _ = collect_candidate_pool(original, execute, info, pool_limit=POOL_LIMIT)
        except Exception as e:
            print(f"skip {rid} ({original}): {e}", flush=True)
            continue

        if not pool:
            continue

        candidates = []
        for uri, c in pool.items():
            label = c["labels"].get("en") or next(iter(c["labels"].values()), None)
            candidates.append({"uri": uri, "label": label, "description": c.get("description")})

        gold_in_pool = gold in pool
        if not gold_in_pool:
            gp = get_entity_profile(gold, execute, lang="en")
            candidates.append({
                "uri": gold,
                "label": (gp or {}).get("label"),
                "description": (gp or {}).get("description"),
            })

        op = get_entity_profile(original, execute, lang="en")
        gp = get_entity_profile(gold, execute, lang="en")

        seen_original.add(original)
        item = {
            "id": rid,
            "language": lang,
            "query": nquery,
            "original": original,
            "original_label": (op or {}).get("label"),
            "original_description": (op or {}).get("description"),
            "gold": gold,
            "gold_label": (gp or {}).get("label"),
            "gold_description": (gp or {}).get("description"),
            "gold_in_pool": gold_in_pool,
            "n_candidates": len(candidates),
            "candidates": candidates,
            "conditions": info["conditions"][original],
        }
        eval_items.append(item)
        node_uris.add(original)
        node_uris.add(gold)
        node_uris.update(c["uri"] for c in candidates)
        print(f"[{len(eval_items)}/{N}] {rid} {original}->{gold} "
              f"cands={len(candidates)} gold_in_pool={gold_in_pool}", flush=True)

    json.dump(eval_items, open(OUT / "eval_set.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(sorted(node_uris), open(OUT / "node_uris.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    retrieved = sum(1 for it in eval_items if it["gold_in_pool"])
    print(f"\nWrote {len(eval_items)} eval items, {len(node_uris)} unique nodes.")
    print(f"Gold naturally retrieved into pool: {retrieved}/{len(eval_items)} "
          f"({retrieved/len(eval_items):.1%}) at pool_limit={POOL_LIMIT}.")


if __name__ == "__main__":
    main()

"""
Stratified, per-language variant of `_prepare_eval_data.py`.

The original script samples N items from a single global shuffle of gold pairs
across all languages, which (given the benchmarks' natural language mix) yields
an English-heavy set. This script instead samples EVAL_N_PER_LANG items
independently for EACH of the 5 supported languages (en, de, fr, ru, uk), so
per-language performance can be compared on equal footing.

Output schema is byte-identical to `_prepare_eval_data.py`'s eval_set.json /
node_uris.json (each item already carries a "language" field), so the existing
run_experiment.py / eval_common.py harnesses work unchanged against the result.

Run from the DynBench-Backend root:
    EVAL_N_PER_LANG=100 EXP_DATA_DIR=data_lang100 \
        python3 experiments/rdf2vec-optimizations/_prepare_eval_data_stratified.py
"""
import json
import os
import random
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # DynBench-Backend root

from utils.sparql import execute as raw_execute, normal_sparql
from utils.wikidata import (
    collect_candidate_pool, get_entity_profile, WIKIDATA_PREFIX,
)
from tests.test_method_comparison import build_info, LIVE_ENDPOINT, LIVE_AGENT, make_caching_execute
from _prepare_eval_data import gold_pairs  # noqa: E402  (reuse the exact diff logic)

LANGUAGES = ["en", "de", "fr", "ru", "uk"]
N_PER_LANG = int(os.environ.get("EVAL_N_PER_LANG", "100"))
POOL_LIMIT = int(os.environ.get("EVAL_POOL_LIMIT", "50"))
SEED = int(os.environ.get("EVAL_SEED", "42"))
QUERY_TIMEOUT = float(os.environ.get("EVAL_QUERY_TIMEOUT", "30.0"))

OUT = Path(__file__).parent / os.environ.get("EXP_DATA_DIR", "data_lang100")
OUT.mkdir(parents=True, exist_ok=True)


def build_item(rid, lang, query, original, gold, execute, seen_original):
    if original in seen_original:
        return None
    nquery = normal_sparql(query, WIKIDATA_PREFIX)
    info = build_info(nquery, original, execute)
    if not info.get("conditions", {}).get(original):
        return None  # no type/query conditions -> no pool can be built
    pool, _ = collect_candidate_pool(original, execute, info, pool_limit=POOL_LIMIT)
    if not pool:
        return None

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

    return {
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


def main():
    raw = partial(raw_execute, endpoint_url=LIVE_ENDPOINT, agent=LIVE_AGENT, delay=0.4, timeout=QUERY_TIMEOUT)
    execute = make_caching_execute(raw, {})

    pairs_by_lang = {lg: [] for lg in LANGUAGES}
    for rid, lang, query, original, gold in gold_pairs():
        if lang in pairs_by_lang:
            pairs_by_lang[lang].append((rid, lang, query, original, gold))
    for lg in LANGUAGES:
        random.Random(SEED).shuffle(pairs_by_lang[lg])
        print(f"{lg}: {len(pairs_by_lang[lg])} candidate gold pairs available", flush=True)

    seen_original = set()  # global dedup across languages too (originals are language-tagged records,
                            # but the same Wikidata entity could recur)
    eval_items = []
    node_uris = set()
    per_lang_count = {lg: 0 for lg in LANGUAGES}

    for lg in LANGUAGES:
        for rid, lang, query, original, gold in pairs_by_lang[lg]:
            if per_lang_count[lg] >= N_PER_LANG:
                break
            try:
                item = build_item(rid, lang, query, original, gold, execute, seen_original)
            except Exception as e:
                print(f"skip {rid} ({original}): {e}", flush=True)
                continue
            if item is None:
                continue
            seen_original.add(original)
            eval_items.append(item)
            per_lang_count[lg] += 1
            node_uris.add(original)
            node_uris.add(gold)
            node_uris.update(c["uri"] for c in item["candidates"])
            print(f"[{lg} {per_lang_count[lg]}/{N_PER_LANG}] (total {len(eval_items)}) "
                  f"{rid} {original}->{gold} cands={item['n_candidates']} "
                  f"gold_in_pool={item['gold_in_pool']}", flush=True)
        if per_lang_count[lg] < N_PER_LANG:
            print(f"WARNING: only found {per_lang_count[lg]}/{N_PER_LANG} usable items for '{lg}' "
                  f"(exhausted {len(pairs_by_lang[lg])} candidate pairs)", flush=True)

    json.dump(eval_items, open(OUT / "eval_set.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(sorted(node_uris), open(OUT / "node_uris.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    retrieved = sum(1 for it in eval_items if it["gold_in_pool"])
    print(f"\nWrote {len(eval_items)} eval items ({per_lang_count}), {len(node_uris)} unique nodes.")
    print(f"Gold naturally retrieved into pool: {retrieved}/{len(eval_items)} "
          f"({retrieved/len(eval_items):.1%}) at pool_limit={POOL_LIMIT}.")


if __name__ == "__main__":
    main()

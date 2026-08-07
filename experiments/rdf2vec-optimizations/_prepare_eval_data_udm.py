"""
Adds Udmurt ("udm") as a 6th language to the existing lang100 stratified eval set.

`benchmarks/QALD_udm.csv` is a Udmurt translation of the (English) QALD questions
that `DynQALD.json`'s `language == "en"` records are drawn from -- matched here by
exact question text (407/407 of DynQALD's en records match a csv row). Since
neither RDF2Vec nor the sentence-embedding method ever reads the question text
(only the SPARQL query's original/gold entity pair and the candidates' English
labels), a "udm" item is mechanically identical to the "en" item for the same
underlying (original, gold) pair -- the language field only changes which bucket
it's reported under.

That matters for sampling: DynBench draws its substitution targets from a fairly
small recurring pool of entities, so of the 406 gold-pairs with a Udmurt
translation, only ~23 have an `original` entity not already claimed by the
existing 478-item, 5-language eval set (en/de/fr/ru/uk). Enforcing the same
global-uniqueness rule the original stratified script used would cap Udmurt at
~23 items, far below the 88-100/language of the others. Per-repo decision: allow
udm items to reuse entities already tested under another language (self-dedup
only within udm) to reach a comparable n=100. This does NOT make udm's results
circular -- ranks are recomputed from scratch against the (extended) local graph
and the (extended) global RDF2Vec corpus, and the fraction of items that happen to
share an entity with another language's item is reported for transparency.

Run from the DynBench-Backend root:
    EVAL_N_PER_LANG=100 EXP_DATA_DIR=data_lang100 EXP_OUT_DIR=data_lang100_udm \
        MAT_QUERY_DELAY=2.0 \
        .venv/bin/python experiments/rdf2vec-optimizations/_prepare_eval_data_udm.py
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # DynBench-Backend root

from utils.sparql import execute as raw_execute, sparql_results_to_list_of_dicts
from utils.wikidata import WIKIDATA_PREFIX
from tests.test_method_comparison import LIVE_ENDPOINT, LIVE_AGENT, make_caching_execute
from _prepare_eval_data_stratified import build_item  # noqa: E402  (reuse pool/candidate logic)

ENT = re.compile(r"wd:Q\d+")

N_PER_LANG = int(os.environ.get("EVAL_N_PER_LANG", "100"))
SEED = int(os.environ.get("EVAL_SEED", "42"))
QUERY_TIMEOUT = float(os.environ.get("EVAL_QUERY_TIMEOUT", "30.0"))
QUERY_DELAY = float(os.environ.get("EVAL_QUERY_DELAY", "2.0"))

IN_DIR = Path(__file__).parent / os.environ.get("EXP_DATA_DIR", "data_lang100")
OUT_DIR = Path(__file__).parent / os.environ.get("EXP_OUT_DIR", "data_lang100_udm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def udm_translations():
    """{english_question: udmurt_translation} for non-empty translations."""
    out = {}
    with open(BENCHMARKS / "QALD_udm.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:
        if len(row) < 3:
            continue
        eng, udm = row[0].strip(), row[2].strip()
        if eng and udm:
            out[eng] = udm
    return out


def udm_gold_pairs():
    """Yield (record_id, query, original_entity, gold_entity) for DynQALD en records
    that are clean single-entity swaps AND have a non-empty Udmurt translation."""
    translations = udm_translations()
    records = json.load(open(BENCHMARKS / "DynQALD.json", encoding="utf-8"))
    for r in records:
        if r.get("language") != "en":
            continue
        if r.get("question", "").strip() not in translations:
            continue
        o, ne = r.get("query", ""), r.get("new query", "")
        so, sne = set(ENT.findall(o)), set(ENT.findall(ne))
        do, dn = so - sne, sne - so
        if len(do) == 1 and len(dn) == 1:
            yield r.get("id"), o, next(iter(do)), next(iter(dn))


def native_label(entity, execute):
    """Best-available native-script label for `entity`: Udmurt if Wikidata has one,
    else Russian, else the language-independent 'mul' label. Udmurt label coverage
    on Wikidata is sparse, so this is purely a display/transparency field alongside
    the (always-English) original_label/gold_label the ranking pipeline actually
    uses -- it does not affect any method's ranking. Returns (label, tier) where
    tier in {"udm", "ru", "mul", None}."""
    query = (
        "SELECT ?udm_label ?ru_label ?mul_label WHERE {"
        f"  OPTIONAL {{ {entity} rdfs:label ?udm_label . FILTER(LANG(?udm_label) = 'udm') }}"
        f"  OPTIONAL {{ {entity} rdfs:label ?ru_label . FILTER(LANG(?ru_label) = 'ru') }}"
        f"  OPTIONAL {{ {entity} rdfs:label ?mul_label . FILTER(LANG(?mul_label) = 'mul') }}"
        "} LIMIT 1"
    )
    try:
        result = sparql_results_to_list_of_dicts(execute(query), WIKIDATA_PREFIX)
        if not result:
            return None, None
        r = result[0]
        if r.get("udm_label"):
            return r["udm_label"], "udm"
        if r.get("ru_label"):
            return r["ru_label"], "ru"
        if r.get("mul_label"):
            return r["mul_label"], "mul"
        return None, None
    except Exception:
        return None, None


def main():
    existing_eval = json.load(open(IN_DIR / "eval_set.json", encoding="utf-8"))
    existing_nodes = set(json.load(open(IN_DIR / "node_uris.json", encoding="utf-8")))
    existing_originals = {it["original"] for it in existing_eval}

    pairs = list(udm_gold_pairs())
    random.Random(SEED).shuffle(pairs)
    print(f"udm: {len(pairs)} candidate gold pairs available "
          f"(of which {sum(1 for p in pairs if p[2] in existing_originals)} share an "
          f"`original` entity already used by en/de/fr/ru/uk)", flush=True)

    raw = partial(raw_execute, endpoint_url=LIVE_ENDPOINT, agent=LIVE_AGENT,
                  delay=QUERY_DELAY, timeout=QUERY_TIMEOUT)
    execute = make_caching_execute(raw, {})

    seen_original = set()  # self-dedup only within udm; overlap with other languages is allowed
    udm_items = []
    new_nodes = set()
    overlap_count = 0
    label_tier_counts = Counter()

    for rid, query, original, gold in pairs:
        if len(udm_items) >= N_PER_LANG:
            break
        try:
            item = build_item(rid, "udm", query, original, gold, execute, seen_original)
        except Exception as e:
            print(f"skip {rid} ({original}): {e}", flush=True)
            continue
        if item is None:
            continue
        item["original_label_native"], orig_tier = native_label(original, execute)
        item["gold_label_native"], gold_tier = native_label(gold, execute)
        label_tier_counts[orig_tier] += 1
        label_tier_counts[gold_tier] += 1
        seen_original.add(original)
        if original in existing_originals:
            overlap_count += 1
        udm_items.append(item)
        new_nodes.add(original)
        new_nodes.add(gold)
        new_nodes.update(c["uri"] for c in item["candidates"])
        print(f"[udm {len(udm_items)}/{N_PER_LANG}] {rid} {original}->{gold} "
              f"cands={item['n_candidates']} gold_in_pool={item['gold_in_pool']} "
              f"overlaps_existing={original in existing_originals} "
              f"native_label_tiers={orig_tier}/{gold_tier}", flush=True)

    if len(udm_items) < N_PER_LANG:
        print(f"WARNING: only found {len(udm_items)}/{N_PER_LANG} usable udm items "
              f"(exhausted {len(pairs)} candidate pairs)", flush=True)

    merged_eval = existing_eval + udm_items
    merged_nodes = sorted(existing_nodes | new_nodes)

    json.dump(merged_eval, open(OUT_DIR / "eval_set.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(merged_nodes, open(OUT_DIR / "node_uris.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(sorted(new_nodes), open(OUT_DIR / "new_node_uris.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    retrieved = sum(1 for it in udm_items if it["gold_in_pool"])
    print(f"\nWrote {len(merged_eval)} total eval items "
          f"({len(existing_eval)} existing + {len(udm_items)} new udm), "
          f"{len(merged_nodes)} unique nodes ({len(new_nodes)} new).")
    print(f"udm gold naturally retrieved into pool: {retrieved}/{len(udm_items)} "
          f"({retrieved/len(udm_items):.1%}).")
    print(f"udm items sharing an `original` with an existing en/de/fr/ru/uk item: "
          f"{overlap_count}/{len(udm_items)} ({overlap_count/len(udm_items):.1%}).")
    total_labels = sum(label_tier_counts.values())
    print(f"native-label tiers across {total_labels} (original+gold) lookups: "
          f"udm={label_tier_counts['udm']} ru={label_tier_counts['ru']} "
          f"mul={label_tier_counts['mul']} none={label_tier_counts[None]}")


if __name__ == "__main__":
    main()

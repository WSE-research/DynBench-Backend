"""
Incremental variant of `_materialize_graph.py`: extends an *existing* local_graph.nt
with only the nodes that aren't already covered by it, instead of re-querying every
node from scratch.

Motivation: adding a 6th language (udm) to the lang100 eval set introduces ~100 new
items, but the existing local_graph.nt already has depth-2 truthy-edge coverage for
the other ~10,615 nodes (362,891 triples, took ~1.8h against the live endpoint to
build). Re-running the original script against the full merged node_uris.json would
re-query all of that for no benefit. This script instead:

  1. Loads the existing graph's distinct subjects (nodes we already have depth-1
     edges for).
  2. Runs depth-1 expansion only for `new_node_uris.json` entries not already a
     subject in the existing graph.
  3. Runs depth-2 expansion only for newly-discovered depth-1 objects not already a
     subject in the existing (extended) graph.
  4. Writes the union (existing triples + newly fetched triples) as the output
     local_graph.nt, so the result is byte-for-byte what a from-scratch run over the
     full merged node set would have produced (same edges_of query, same D1/D2
     limits), just without re-fetching what's already cached on disk.

Run from the DynBench-Backend root:
    EXP_DATA_DIR=data_lang100 EXP_OUT_DIR=data_lang100_udm MAT_QUERY_DELAY=2.0 \
        .venv/bin/python experiments/rdf2vec-optimizations/_materialize_graph_incremental.py
"""
import json
import os
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.sparql import execute as raw_execute, sparql_results_to_list_of_dicts, short2uri
from utils.wikidata import WIKIDATA_PREFIX
from tests.test_method_comparison import LIVE_ENDPOINT, LIVE_AGENT, make_caching_execute

DEPTH = int(os.environ.get("MAT_DEPTH", "2"))
D1_LIMIT = int(os.environ.get("MAT_D1_LIMIT", "500"))
D2_LIMIT = int(os.environ.get("MAT_D2_LIMIT", "200"))
MAX_D2 = int(os.environ.get("MAT_MAX_D2", "4000"))
QUERY_TIMEOUT = float(os.environ.get("MAT_QUERY_TIMEOUT", "20.0"))
QUERY_DELAY = float(os.environ.get("MAT_QUERY_DELAY", "2.0"))

IN_DIR = Path(__file__).parent / os.environ.get("EXP_DATA_DIR", "data_lang100")
OUT_DIR = Path(__file__).parent / os.environ.get("EXP_OUT_DIR", "data_lang100_udm")
ENTITY_NS = "http://www.wikidata.org/entity/Q"
DIRECT_NS = "http://www.wikidata.org/prop/direct/"


def edges_of(node_short, execute, limit):
    q = (
        f"SELECT ?p ?o WHERE {{ {node_short} ?p ?o . "
        f'FILTER(STRSTARTS(STR(?p), "{DIRECT_NS}")) '
        f'FILTER(STRSTARTS(STR(?o), "{ENTITY_NS}")) }} LIMIT {limit}'
    )
    try:
        rows = sparql_results_to_list_of_dicts(execute(q), WIKIDATA_PREFIX)
        return [(r["p"], r["o"]) for r in rows if r.get("p") and r.get("o")]
    except Exception as e:
        print(f"  ! {node_short}: {e}", flush=True)
        return []


def load_existing_triples(nt_path):
    """Parse N-Triples back into (subject_short, pred_short, obj_short) using the
    same wd:/wdt: short-form the rest of the pipeline uses."""
    from utils.sparql import uri2short
    triples = set()
    subjects = set()
    with open(nt_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("<"):
                continue
            parts = line.rstrip(" .\n").split("> <")
            if len(parts) != 3:
                continue
            s = parts[0][1:]
            p = parts[1]
            o = parts[2][:-1]
            s_short = uri2short(s, WIKIDATA_PREFIX)
            p_short = uri2short(p, WIKIDATA_PREFIX)
            o_short = uri2short(o, WIKIDATA_PREFIX)
            triples.add((s_short, p_short, o_short))
            subjects.add(s_short)
    return triples, subjects


def main():
    new_nodes = json.load(open(OUT_DIR / "new_node_uris.json", encoding="utf-8"))
    existing_nt = IN_DIR / "local_graph.nt"

    print(f"Loading existing graph {existing_nt} ...", flush=True)
    t0 = time.time()
    triples, existing_subjects = load_existing_triples(existing_nt)
    print(f"  {len(triples)} triples, {len(existing_subjects)} distinct subjects "
          f"({time.time()-t0:.0f}s)", flush=True)

    raw = partial(raw_execute, endpoint_url=LIVE_ENDPOINT, agent=LIVE_AGENT, delay=QUERY_DELAY, timeout=QUERY_TIMEOUT)
    execute = make_caching_execute(raw, {}, max_size=200000)

    todo_d1 = [n for n in new_nodes if n not in existing_subjects]
    print(f"Depth-1: {len(new_nodes)} new nodes, {len(todo_d1)} not already covered ...", flush=True)
    t0 = time.time()
    d1_objects = Counter()
    for i, node in enumerate(todo_d1, 1):
        for p, o in edges_of(node, execute, D1_LIMIT):
            triples.add((node, p, o))
            d1_objects[o] += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(todo_d1)}] triples={len(triples)} d1_objs={len(d1_objects)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    all_subjects_now = existing_subjects | set(todo_d1)

    if DEPTH >= 2:
        d2_candidates = [o for o, _ in d1_objects.most_common() if o not in all_subjects_now][:MAX_D2]
        print(f"Depth-2: expanding {len(d2_candidates)} new object entities "
              f"(of {len(d1_objects)} unique; cap {MAX_D2}) ...", flush=True)
        t0 = time.time()
        for i, node in enumerate(d2_candidates, 1):
            for p, o in edges_of(node, execute, D2_LIMIT):
                triples.add((node, p, o))
            if i % 100 == 0:
                print(f"  [{i}/{len(d2_candidates)}] triples={len(triples)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    out = OUT_DIR / "local_graph.nt"
    with open(out, "w", encoding="utf-8") as f:
        for s, p, o in sorted(triples):
            f.write(f"<{short2uri(s, WIKIDATA_PREFIX)}> "
                    f"<{short2uri(p, WIKIDATA_PREFIX)}> "
                    f"<{short2uri(o, WIKIDATA_PREFIX)}> .\n")

    subjects = {s for s, _, _ in triples}
    objects = {o for _, _, o in triples}
    all_nodes = json.load(open(OUT_DIR / "node_uris.json", encoding="utf-8"))
    json.dump(
        {"nodes_requested": len(all_nodes), "depth": DEPTH, "triples": len(triples),
         "distinct_subjects": len(subjects), "distinct_objects": len(objects),
         "d1_unique_objects": len(d1_objects), "max_d2": MAX_D2,
         "incremental_from": str(existing_nt), "new_nodes_fetched": len(todo_d1)},
        open(OUT_DIR / "local_graph_stats.json", "w"), indent=2)
    print(f"\nWrote {len(triples)} triples to {out}")
    print(f"distinct subjects={len(subjects)} objects={len(objects)}")


if __name__ == "__main__":
    main()

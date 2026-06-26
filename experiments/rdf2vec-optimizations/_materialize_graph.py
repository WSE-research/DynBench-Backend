"""
Materialize a bounded local knowledge graph for the RDF2Vec optimization
experiments, so the parameter sweeps run offline (fast, reproducible, and free
of the live-endpoint walk-extraction timeouts that crippled RDF2Vec coverage in
the original three-way comparison).

Graph scope (Wikidata "truthy" entity sub-graph):
  - depth 1: for every evaluation node (originals, gold substitutes, candidates),
    all  wdt:  edges whose object is a Wikidata entity (wd:Q...). Literal-valued
    statements are dropped so the walks capture structural / type similarity.
  - depth 2: the same edges for the depth-1 object entities (the shared types and
    neighbours that create cross-candidate similarity), bounded by MAX_D2 and a
    per-node LIMIT to keep the graph tractable.

The result is written as N-Triples (local_graph.nt) which pyRDF2Vec's KG reads
via rdflib; RDF2VecEmbedder(local_graph.nt, ...) then walks it with no network.

Run from the DynBench-Backend root with its venv (queries Wikidata once):
    .venv/bin/python experiments/rdf2vec-optimizations/_materialize_graph.py
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

DATA = Path(__file__).parent / os.environ.get("EXP_DATA_DIR", "data")
ENTITY_NS = "http://www.wikidata.org/entity/Q"
DIRECT_NS = "http://www.wikidata.org/prop/direct/"


def edges_of(node_short, execute, limit):
    """Return list of (p_short, o_short) truthy edges with entity objects for node."""
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


def main():
    nodes = json.load(open(DATA / "node_uris.json", encoding="utf-8"))
    raw = partial(raw_execute, endpoint_url=LIVE_ENDPOINT, agent=LIVE_AGENT, delay=0.3, timeout=30.0)
    execute = make_caching_execute(raw, {}, max_size=200000)

    triples = set()
    d1_objects = Counter()

    print(f"Depth-1: {len(nodes)} nodes ...", flush=True)
    t0 = time.time()
    for i, node in enumerate(nodes, 1):
        for p, o in edges_of(node, execute, D1_LIMIT):
            triples.add((node, p, o))
            d1_objects[o] += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(nodes)}] triples={len(triples)} d1_objs={len(d1_objects)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if DEPTH >= 2:
        already = set(nodes)
        # expand the shared neighbours first (they carry the similarity signal)
        d2_nodes = [o for o, _ in d1_objects.most_common() if o not in already][:MAX_D2]
        print(f"Depth-2: expanding {len(d2_nodes)} object entities "
              f"(of {len(d1_objects)} unique; cap {MAX_D2}) ...", flush=True)
        t0 = time.time()
        for i, node in enumerate(d2_nodes, 1):
            for p, o in edges_of(node, execute, D2_LIMIT):
                triples.add((node, p, o))
            if i % 100 == 0:
                print(f"  [{i}/{len(d2_nodes)}] triples={len(triples)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    out = DATA / "local_graph.nt"
    with open(out, "w", encoding="utf-8") as f:
        for s, p, o in sorted(triples):
            f.write(f"<{short2uri(s, WIKIDATA_PREFIX)}> "
                    f"<{short2uri(p, WIKIDATA_PREFIX)}> "
                    f"<{short2uri(o, WIKIDATA_PREFIX)}> .\n")

    subjects = {s for s, _, _ in triples}
    objects = {o for _, _, o in triples}
    json.dump(
        {"nodes_requested": len(nodes), "depth": DEPTH, "triples": len(triples),
         "distinct_subjects": len(subjects), "distinct_objects": len(objects),
         "d1_unique_objects": len(d1_objects), "max_d2": MAX_D2},
        open(DATA / "local_graph_stats.json", "w"), indent=2)
    print(f"\nWrote {len(triples)} triples to {out}")
    print(f"distinct subjects={len(subjects)} objects={len(objects)}")


if __name__ == "__main__":
    main()

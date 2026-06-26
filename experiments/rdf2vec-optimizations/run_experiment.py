"""
RDF2Vec optimization experiment runner (offline, on the materialized local graph).

For each configuration we rank every evaluation item's candidate set by similarity
to the original entity and record where the GOLD substitute lands. Primary metrics
are Hits@k, MRR and the rank of the gold. Because the gold is a single,
popularity-matched substitute (see the strategy reports), absolute numbers are
only meaningful RELATIVE to the calibrating sentence-embedding reference and the
random baseline, both reported alongside every RDF2Vec config.

Modes:
  - per_request : train one Word2Vec per item on that item's (original+candidates)
                  walks  -- this is exactly how the production pipeline uses RDF2Vec.
  - global      : train ONE Word2Vec on the walks of ALL evaluation nodes, then look
                  up vectors -- tests the "tiny per-request corpus" hypothesis.
  - embedding   : sentence-transformer reference (text of "label, description").

Usage:
    .venv/bin/python experiments/rdf2vec-optimizations/run_experiment.py CONFIG [CONFIG ...]
    .venv/bin/python experiments/rdf2vec-optimizations/run_experiment.py --all
    .venv/bin/python experiments/rdf2vec-optimizations/run_experiment.py --list
"""
import json
import math
import os
import sys
import time
from pathlib import Path

# Reproducibility: CPython salts str/bytes hashing per process (PYTHONHASHSEED), which
# changes set-iteration order during RDF2Vec walk extraction and therefore the walks,
# the trained vectors and the gold ranks. (The crc32 Word2Vec hashfxn only pins
# gensim's side, not this.) Without it two runs -- even two *sequential* ones -- of the
# same config disagree, and parallel workers each get their own seed. Pin it for the
# whole run (parent + all forked workers) so every result is bit-identical and
# reproducible regardless of --jobs. It must be set before the interpreter starts, so
# re-exec once if unset/wrong.
_WANT_HASHSEED = "0"
if os.environ.get("PYTHONHASHSEED") != _WANT_HASHSEED:
    os.environ["PYTHONHASHSEED"] = _WANT_HASHSEED
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Keep each worker process single-threaded for BLAS / torch so that running many
# configs in parallel (--jobs / EXP_JOBS) uses one core each instead of having 20
# processes each spawn 20 native threads and thrash. Must be set before numpy and
# torch import their native thread pools, hence at the very top of the module.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.sparql import short2uri
from utils.wikidata import WIKIDATA_PREFIX, entity_text  # entity_text re-exported via wikidata? fallback below

HERE = Path(__file__).parent
DATA = HERE / os.environ.get("EXP_DATA_DIR", "data")
RESULTS = HERE / os.environ.get("EXP_RESULTS_DIR", "results")
RESULTS.mkdir(exist_ok=True)
GRAPH = str(DATA / "local_graph.nt")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42
REPEATS = int(os.environ.get("EXP_REPEATS", "5"))  # stochastic repeats (seeds 0..REPEATS-1)

try:
    from utils.embeddings import entity_text as _entity_text
    entity_text = _entity_text
except Exception:
    def entity_text(label, description=None):
        if not label:
            return None
        return f"{label.strip()}, {description.strip()}" if description else label.strip()


# --------------------------------------------------------------------------- #
# configurations
# --------------------------------------------------------------------------- #
def r2v(**kw):
    base = dict(mode="per_request", walker="random", sampler="uniform",
                max_depth=2, max_walks=4, vector_size=100, epochs=10,
                window=5, sg=1, min_count=1, negative=5)
    base.update(kw)
    return base


CONFIGS = {
    # baseline reproduction of the production settings
    "baseline_per_request": r2v(max_depth=2, max_walks=4),

    # --- walk budget sweep ---
    "walks_10": r2v(max_walks=10),
    "walks_25": r2v(max_walks=25),
    "walks_50": r2v(max_walks=50),
    "walks_100": r2v(max_walks=100),
    "walks_200": r2v(max_walks=200),
    "depth_1": r2v(max_depth=1, max_walks=50),
    "depth_3_walks_50": r2v(max_depth=3, max_walks=50),
    "depth_4_walks_50": r2v(max_depth=4, max_walks=50),

    # --- Word2Vec hyperparameters (on a generous walk budget) ---
    "vec_200": r2v(max_walks=100, vector_size=200),
    "vec_300": r2v(max_walks=100, vector_size=300),
    "epochs_50": r2v(max_walks=100, epochs=50),
    "epochs_100": r2v(max_walks=100, epochs=100),
    "window_10": r2v(max_walks=100, window=10),
    "cbow": r2v(max_walks=100, sg=0),

    # --- the key hypothesis: tiny per-request corpus vs one global model ---
    "global_baseline": r2v(mode="global", max_depth=2, max_walks=4),
    "global_walks_100": r2v(mode="global", max_walks=100),
    "global_depth3_walks_200": r2v(mode="global", max_depth=3, max_walks=200, epochs=50, vector_size=200),
    # train on EVERY entity in the graph (largest local corpus ~ proper RDF2Vec usage)
    "global_full_corpus": r2v(mode="global", train_on="all", max_walks=100, vector_size=200, epochs=50),
    "global_full_corpus_deep": r2v(mode="global", train_on="all", max_depth=3, max_walks=200,
                                   vector_size=200, epochs=100, window=10),

    # --- walker strategy (generous budget, global corpus) ---
    "walker_wl": r2v(mode="global", walker="weisfeiler_lehman", max_walks=100),
    "walker_halk": r2v(mode="global", walker="halk", max_walks=100),
    "walker_ngram": r2v(mode="global", walker="ngram", max_walks=100),
    "walker_walklet": r2v(mode="global", walker="walklet", max_walks=100),

    # --- sampler strategy ---
    "sampler_pagerank": r2v(mode="global", sampler="pagerank", max_walks=100),
    "sampler_objfreq": r2v(mode="global", sampler="objfreq", max_walks=100),
    "sampler_wide": r2v(mode="global", sampler="wide", max_walks=100),

    # --- depth-1 is the strongest lever; test it combined with global training ---
    "global_depth1": r2v(mode="global", max_depth=1, max_walks=100),
    "global_depth1_fullcorpus": r2v(mode="global", train_on="all", max_depth=1, max_walks=100),
    "depth1_walks_200": r2v(max_depth=1, max_walks=200),

    # --- best-effort combined config ---
    "combined_best": r2v(mode="global", walker="random", sampler="pagerank",
                         max_depth=3, max_walks=200, vector_size=200, epochs=100, window=10),

    # reference + baselines
    "embedding_reference": {"mode": "embedding"},
}


# --------------------------------------------------------------------------- #
# pyRDF2Vec helpers (local graph, configurable components)
# --------------------------------------------------------------------------- #
def stable_hash(token: str) -> int:
    import zlib
    return zlib.crc32(token.encode("utf-8"))


def make_walker(cfg, seed):
    from pyrdf2vec.walkers import (RandomWalker, WLWalker, HALKWalker,
                                   NGramWalker, WalkletWalker)
    from pyrdf2vec import samplers as S
    sampler_map = {
        "uniform": S.UniformSampler,
        "pagerank": S.PageRankSampler,
        "objfreq": getattr(S, "ObjFreqSampler", S.UniformSampler),
        "wide": getattr(S, "WideSampler", S.UniformSampler),
    }
    sampler = sampler_map[cfg["sampler"]]()
    common = dict(max_depth=cfg["max_depth"], max_walks=cfg["max_walks"],
                  n_jobs=1, random_state=seed, sampler=sampler)
    walker_map = {
        "random": RandomWalker, "weisfeiler_lehman": WLWalker, "halk": HALKWalker,
        "ngram": NGramWalker, "walklet": WalkletWalker,
    }
    return walker_map[cfg["walker"]](**common)


def embed_uris(uris, cfg, kg, seed):
    """Train one Word2Vec on the walks of `uris` and return {uri: L2-normalized vector}."""
    from pyrdf2vec import RDF2VecTransformer
    from pyrdf2vec.embedders import Word2Vec
    w2v = Word2Vec(vector_size=cfg["vector_size"], epochs=cfg["epochs"],
                   sg=cfg["sg"], sample=0, workers=1, seed=seed,
                   hashfxn=stable_hash, window=cfg["window"],
                   min_count=cfg["min_count"], negative=cfg["negative"])
    transformer = RDF2VecTransformer(w2v, walkers=[make_walker(cfg, seed)], verbose=0)
    vectors, _ = transformer.fit_transform(kg, list(uris))
    out = {}
    for u, v in zip(uris, vectors):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        out[u] = v / n if n else v
    return out


def load_kg():
    from pyrdf2vec.graphs import KG
    return KG(GRAPH)


# --------------------------------------------------------------------------- #
# ranking + metrics
# --------------------------------------------------------------------------- #
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


def run_rdf2vec(cfg, seed):
    kg = load_kg()
    subjects = graph_subjects()
    full = lambda s: short2uri(s, WIKIDATA_PREFIX)
    items = EVAL

    global_vecs = None
    if cfg["mode"] == "global":
        if cfg.get("train_on") == "all":
            # train on EVERY entity in the materialized graph (largest possible
            # corpus locally) -- the closest internal proxy to "proper" large-corpus
            # RDF2Vec, and the strongest test of the tiny-corpus hypothesis.
            all_nodes = sorted(subjects)
        else:
            all_nodes = sorted({full(it["original"]) for it in items} |
                               {full(c["uri"]) for it in items for c in it["candidates"]})
            all_nodes = [u for u in all_nodes if u in subjects]
        global_vecs = embed_uris(all_nodes, cfg, kg, seed)

    per_item = []
    for it in items:
        orig = full(it["original"])
        cand_full = {c["uri"]: full(c["uri"]) for c in it["candidates"]}
        if cfg["mode"] == "global":
            vecs = global_vecs
        else:
            uris = [orig] + [u for u in cand_full.values() if u in subjects]
            uris = sorted(set(u for u in uris if u in subjects))
            vecs = embed_uris(uris, cfg, kg, seed) if orig in subjects and len(uris) > 1 else {}
        ov = vecs.get(orig)
        sim_of = {}
        if ov is not None:
            for short, furi in cand_full.items():
                cv = vecs.get(furi)
                if cv is not None and furi != orig:
                    sim_of[short] = float(ov @ cv)
        rank, n = gold_rank_from_sims(it, sim_of)
        per_item.append({"id": it["id"], "gold_rank": rank, "n_ranked": n,
                         "n_candidates": len(it["candidates"])})
    return per_item


def run_embedding():
    from utils.embeddings import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder(EMBEDDING_MODEL)
    per_item = []
    for it in EVAL:
        otext = entity_text(it.get("original_label"), it.get("original_description"))
        cand_texts = {c["uri"]: entity_text(c.get("label"), c.get("description")) for c in it["candidates"]}
        usable = {u: t for u, t in cand_texts.items() if t}
        if not otext or not usable:
            per_item.append({"id": it["id"], "gold_rank": None, "n_ranked": 0,
                             "n_candidates": len(it["candidates"])})
            continue
        uris = list(usable.keys())
        vecs = emb.encode([otext] + [usable[u] for u in uris])
        ov = vecs[0]
        sim_of = {u: float(ov @ vecs[i + 1]) for i, u in enumerate(uris)}
        rank, n = gold_rank_from_sims(it, sim_of)
        per_item.append({"id": it["id"], "gold_rank": rank, "n_ranked": n,
                         "n_candidates": len(it["candidates"])})
    return per_item


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


def aggregate(name, reps, seconds):
    """Aggregate a config's per-repeat per-item ranks into the summary + result file.
    `seconds` is the total compute time for the config (sum of its repeats), so the
    number is identical whether the repeats ran sequentially or in parallel."""
    import statistics as stx
    cfg = CONFIGS[name]
    nrep = len(reps)
    met = [metrics_one(pi) for pi in reps]

    def mean_std(key):
        vals = [m[key] for m in met]
        return round(stx.mean(vals), 3), (round(stx.pstdev(vals), 3) if nrep > 1 else 0.0)

    mrr_mean, mrr_std = mean_std("MRR")
    rmrr, rh5 = random_baseline(reps[0])

    # per-item mean reciprocal rank across repeats (averages out run noise -> bootstrap over items)
    ids = [p["id"] for p in reps[0]]
    per_item_mean = []
    for idx, iid in enumerate(ids):
        rrs = [(1.0 / pi[idx]["gold_rank"] if pi[idx]["gold_rank"] else 0.0) for pi in reps]
        per_item_mean.append({"id": iid, "mean_rr": sum(rrs) / len(rrs),
                              "n_candidates": reps[0][idx]["n_candidates"],
                              "found_frac": sum(1 for pi in reps if pi[idx]["gold_rank"]) / len(reps)})

    summary = {
        "config": name, "mode": cfg["mode"], "repeats": nrep, "items": len(reps[0]),
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
    json.dump({"summary": summary, "config": cfg, "per_repeat_metrics": met,
               "per_item_mean": per_item_mean, "per_repeat": reps},
              open(RESULTS / f"{name}.json", "w"), indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_config(name):
    """Sequential path: compute a config's repeats one after another, then aggregate."""
    cfg = CONFIGS[name]
    t0 = time.time()
    if cfg["mode"] == "embedding":
        reps = [run_embedding()]                       # deterministic -> single repeat
    else:
        reps = [run_rdf2vec(cfg, s) for s in range(REPEATS)]
    return aggregate(name, reps, time.time() - t0)


# --------------------------------------------------------------------------- #
# parallel execution: one process per (config, seed) repeat
# --------------------------------------------------------------------------- #
def _task(args):
    """One unit of parallel work: a single repeat. Runs single-threaded and
    deterministically (workers=1, fixed seed), so each result is bit-identical to
    the sequential path -- parallelism only changes scheduling, never the numbers."""
    name, seed = args
    _ensure_loaded()
    cfg = CONFIGS[name]
    t0 = time.time()
    per_item = run_embedding() if cfg["mode"] == "embedding" else run_rdf2vec(cfg, seed)
    return name, seed, per_item, time.time() - t0


def run_parallel(names, jobs):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    tasks = []
    for name in names:
        if CONFIGS[name]["mode"] == "embedding":
            tasks.append((name, None))                  # deterministic -> single repeat
        else:
            tasks.extend((name, s) for s in range(REPEATS))

    # heaviest tasks first so long stragglers start early -> better core utilisation
    def cost(t):
        c = CONFIGS[t[0]]
        if c["mode"] == "embedding":
            return 1
        return c.get("max_walks", 4) * c.get("epochs", 10) * c.get("max_depth", 2)
    tasks.sort(key=cost, reverse=True)

    reps = {n: {} for n in names}
    secs = {n: 0.0 for n in names}
    failed = set()
    total = len(tasks)
    done = 0
    print(f"# {len(names)} configs -> {total} tasks on {jobs} workers", flush=True)
    ctx = mp.get_context("fork")            # fork: workers inherit EVAL + graph already in RAM
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                             initializer=_ensure_loaded) as ex:
        futs = {ex.submit(_task, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            done += 1
            try:
                name, seed, per_item, elapsed = fut.result()
            except Exception as e:
                failed.add(t[0])
                print(f"# [{done}/{total}] FAILED {t}: {e}", flush=True)
                continue
            reps[name][seed] = per_item
            secs[name] += elapsed
            print(f"# [{done}/{total}] {name} seed={seed} {elapsed:.1f}s", flush=True)

    for name in names:
        if name in failed:
            print(f"# skip aggregate for failed config: {name}", file=sys.stderr)
            continue
        d = reps[name]
        ordered = [d[None]] if CONFIGS[name]["mode"] == "embedding" else [d[s] for s in range(REPEATS)]
        aggregate(name, ordered, secs[name])


# lazily loaded globals
EVAL = None
_SUBJECTS = None


def graph_subjects():
    global _SUBJECTS
    if _SUBJECTS is None:
        subs = set()
        with open(GRAPH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("<"):
                    subs.add(line[1:line.index(">")])
        _SUBJECTS = subs
    return _SUBJECTS


def _ensure_loaded():
    """Load the eval set and graph subjects into this process's globals (idempotent).
    Called in the parent before forking, so fork workers inherit them for free; the
    pool initializer re-runs it as a safety net for non-fork start methods."""
    global EVAL
    if EVAL is None:
        EVAL = json.load(open(DATA / "eval_set.json", encoding="utf-8"))
    graph_subjects()


def main():
    args = sys.argv[1:]

    # --jobs N (or EXP_JOBS env) selects how many (config, seed) repeats run at once.
    jobs = int(os.environ.get("EXP_JOBS", "1"))
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        del args[i:i + 2]

    if not args or args[0] == "--list":
        print("\n".join(CONFIGS.keys()))
        return

    names = list(CONFIGS.keys()) if args[0] == "--all" else args
    unknown = [n for n in names if n not in CONFIGS]
    for n in unknown:
        print(f"unknown config: {n}", file=sys.stderr)
    names = [n for n in names if n in CONFIGS]
    if not names:
        return

    _ensure_loaded()
    if jobs and jobs > 1:
        run_parallel(names, jobs)
    else:
        for name in names:
            try:
                run_config(name)
            except Exception as e:
                import traceback
                print(f"FAILED {name}: {e}", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Sentence-embedding optimization experiment for substitute ranking (EXTENDED sweep).

Mirrors experiments/rdf2vec-optimizations/run_experiment.py: SAME eval sets (read
from ../rdf2vec-optimizations/data and data_n150), SAME metric (imported from
eval_common, byte-identical to the RDF2Vec study), SAME result JSON schema and the
SAME `_analyze.py`. Only the ranking representation differs: candidates are ranked
by similarity of sentence embeddings instead of graph-walk vectors.

Each configuration varies one lever of the embedding method:
  - model     : 14 sentence-transformers models (multilingual, English, E5, large)
  - template  : text fed to the model (label / label+desc / desc / joiners / order)
  - similarity: cosine (L2-normalized dot) / dot (un-normalized) / euclidean
  - prefix    : query/passage/instruction prompts for instruction models

The baseline `model__mMiniLM_baseline` (paraphrase-multilingual-MiniLM-L12-v2,
"label, description", cosine, no prefix) is exactly the production embedding setting
and reproduces the `embedding_reference` in the RDF2Vec results.

Usage:
    .venv/bin/python experiments/embedding-optimizations/run_experiment.py --all [--jobs N]
    .venv/bin/python experiments/embedding-optimizations/run_experiment.py --list
    EXP_DATA_DIR=data_n150 EXP_RESULTS_DIR=results_n150 ... --all
"""
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))                    # for eval_common
import eval_common  # noqa: E402
from eval_common import gold_rank_from_sims, load_eval  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE.parent / "rdf2vec-optimizations" / os.environ.get("EXP_DATA_DIR", "data")
RESULTS = HERE / os.environ.get("EXP_RESULTS_DIR", "results")
EVAL_FILE = DATA / "eval_set.json"

# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
MODELS = {
    # multilingual small/base
    "mMiniLM":         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # baseline
    "mMPNet":          "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "distiluse":       "sentence-transformers/distiluse-base-multilingual-cased-v2",
    # strong English (entity text is English)
    "enMPNet":         "sentence-transformers/all-mpnet-base-v2",
    "enMiniLM":        "sentence-transformers/all-MiniLM-L6-v2",
    # E5 multilingual (instruction; need a prefix)
    "e5small":         "intfloat/multilingual-e5-small",
    "e5base":          "intfloat/multilingual-e5-base",
    "e5large":         "intfloat/multilingual-e5-large",
    # large / heavy (extended set)
    "bgem3":           "BAAI/bge-m3",
    "LaBSE":           "sentence-transformers/LaBSE",
    "e5largeV2":       "intfloat/e5-large-v2",                       # English E5
    "e5largeInstruct": "intfloat/multilingual-e5-large-instruct",
    "bgeLargeEn":      "BAAI/bge-large-en-v1.5",
    "mxbai":           "mixedbread-ai/mxbai-embed-large-v1",
}

# prompts / prefixes
E5Q = "query: "                                                     # E5 (symmetric) prefix
INSTRUCT = ("Instruct: Retrieve entities of the same type and comparable prominence.\n"
            "Query: ")                                              # e5-instruct: query side only
BGE_PROMPT = "Represent this sentence for searching relevant passages: "  # bge/mxbai query prompt

# model-default (query, doc) prefixes; everything else is ("", "")
MODEL_PREFIX = {
    "e5small": (E5Q, E5Q), "e5base": (E5Q, E5Q), "e5large": (E5Q, E5Q),
    "e5largeV2": (E5Q, E5Q), "e5largeInstruct": (INSTRUCT, ""),
}


def cfg(model, template="label_desc", similarity="cosine", pq=None, pd=None):
    dq, dd = MODEL_PREFIX.get(model, ("", ""))
    return {"mode": model, "model": MODELS[model], "template": template,
            "similarity": similarity,
            "prefix_query": dq if pq is None else pq,
            "prefix_doc": dd if pd is None else pd}


# strong models that get the {label, desc} template grid (label_desc is in the model sweep)
GRID_MODELS = ["e5large", "bgem3", "mxbai", "bgeLargeEn", "e5largeInstruct",
               "distiluse", "e5base", "enMPNet"]

CONFIGS = {}
# --- model sweep (template=label_desc, cosine, model-default prefix) ---
for _m in MODELS:
    _name = "model__mMiniLM_baseline" if _m == "mMiniLM" else f"model__{_m}"
    CONFIGS[_name] = cfg(_m)
# --- text-template grid {label, desc} on the strong models ---
for _m in GRID_MODELS:
    CONFIGS[f"tmpl__{_m}_label"] = cfg(_m, template="label")
    CONFIGS[f"tmpl__{_m}_desc"] = cfg(_m, template="desc")
# --- full template sweep on the baseline model (its template lever) ---
CONFIGS["tmpl__mMiniLM_label"] = cfg("mMiniLM", template="label")
CONFIGS["tmpl__mMiniLM_desc"] = cfg("mMiniLM", template="desc")
CONFIGS["tmpl__mMiniLM_label_dot_desc"] = cfg("mMiniLM", template="label_dot_desc")
CONFIGS["tmpl__mMiniLM_label_paren_desc"] = cfg("mMiniLM", template="label_paren_desc")
# --- text-order variant on baseline + top model ---
CONFIGS["tmpl__mMiniLM_desc_label"] = cfg("mMiniLM", template="desc_label")
CONFIGS["tmpl__e5large_desc_label"] = cfg("e5large", template="desc_label")
# --- similarity / normalization sweep (baseline + top model) ---
CONFIGS["sim__mMiniLM_dot"] = cfg("mMiniLM", similarity="dot")
CONFIGS["sim__mMiniLM_euclidean"] = cfg("mMiniLM", similarity="euclidean")
CONFIGS["sim__e5large_dot"] = cfg("e5large", similarity="dot")
CONFIGS["sim__e5large_euclidean"] = cfg("e5large", similarity="euclidean")
# --- prompt / prefix ablations ---
CONFIGS["prefix__e5base_none"] = cfg("e5base", pq="", pd="")
CONFIGS["prefix__e5base_querypassage"] = cfg("e5base", pq="query: ", pd="passage: ")
CONFIGS["prefix__bgeLargeEn_prompt"] = cfg("bgeLargeEn", pq=BGE_PROMPT, pd="")
CONFIGS["prefix__mxbai_prompt"] = cfg("mxbai", pq=BGE_PROMPT, pd="")
CONFIGS["prefix__e5largeInstruct_none"] = cfg("e5largeInstruct", pq="", pd="")


# --------------------------------------------------------------------------- #
# embedding helpers
# --------------------------------------------------------------------------- #
_MODELS = {}


def get_model(hf_id):
    if hf_id not in _MODELS:
        from sentence_transformers import SentenceTransformer
        print(f"# loading {hf_id} ...", flush=True)
        _MODELS[hf_id] = SentenceTransformer(hf_id, device="cpu")
    return _MODELS[hf_id]


def build_text(template, label, desc):
    """Text representation of an entity. `label_desc` matches utils.embeddings.entity_text."""
    label = (label or "").strip()
    desc = (desc or "").strip()
    if template == "desc":
        return desc or None
    if not label:
        return None
    if template == "label":
        return label
    if not desc:
        return label
    if template == "label_desc":
        return f"{label}, {desc}"
    if template == "label_dot_desc":
        return f"{label}. {desc}"
    if template == "label_paren_desc":
        return f"{label} ({desc})"
    if template == "desc_label":
        return f"{desc}, {label}"
    raise ValueError(f"unknown template {template}")


def run_embedding(name):
    c = CONFIGS[name]
    model, template, sim = c["model"], c["template"], c["similarity"]
    pq, pd = c["prefix_query"], c["prefix_doc"]
    normalize = (sim == "cosine")
    m = get_model(model)

    per_item = []
    for it in EVAL:
        otext = build_text(template, it.get("original_label"), it.get("original_description"))
        cand_texts = {ca["uri"]: build_text(template, ca.get("label"), ca.get("description"))
                      for ca in it["candidates"]}
        usable = {u: t for u, t in cand_texts.items() if t}
        if not otext or not usable:
            per_item.append({"id": it["id"], "gold_rank": None, "n_ranked": 0,
                             "n_candidates": len(it["candidates"])})
            continue
        uris = list(usable.keys())
        q = f"{pq}{otext}" if pq else otext
        docs = [(f"{pd}{usable[u]}" if pd else usable[u]) for u in uris]
        vecs = np.asarray(m.encode([q] + docs, normalize_embeddings=normalize,
                                   show_progress_bar=False), dtype=float)
        ov = vecs[0]
        if sim == "euclidean":
            sim_of = {u: -float(np.linalg.norm(ov - vecs[i + 1])) for i, u in enumerate(uris)}
        else:  # cosine (normalized) or dot (un-normalized) -> rank by dot product
            sim_of = {u: float(ov @ vecs[i + 1]) for i, u in enumerate(uris)}
        rank, n = gold_rank_from_sims(it, sim_of)
        per_item.append({"id": it["id"], "gold_rank": rank, "n_ranked": n,
                         "n_candidates": len(it["candidates"])})
    return per_item


def run_config(name):
    t0 = time.time()
    reps = [run_embedding(name)]          # deterministic at inference -> single repeat
    return eval_common.aggregate(RESULTS, name, CONFIGS[name]["mode"], CONFIGS[name],
                                 reps, time.time() - t0)


# --------------------------------------------------------------------------- #
# parallel execution: one worker per MODEL (loads the model once, runs its configs)
# --------------------------------------------------------------------------- #
EVAL = None


def _ensure_eval():
    global EVAL
    if EVAL is None:
        EVAL = load_eval(EVAL_FILE)


def _run_group(names):
    _ensure_eval()
    out = []
    for n in names:
        out.append(run_config(n))
    return out


def run_parallel(names, jobs):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    groups = {}
    for n in names:
        groups.setdefault(CONFIGS[n]["model"], []).append(n)
    tasks = sorted(groups.values(), key=len, reverse=True)
    jobs = min(jobs, len(tasks))
    print(f"# {len(names)} configs over {len(tasks)} models on {jobs} workers", flush=True)
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_ensure_eval) as ex:
        futs = [ex.submit(_run_group, g) for g in tasks]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                import traceback
                print(f"# GROUP FAILED: {e}", flush=True)
                traceback.print_exc()


def main():
    args = sys.argv[1:]
    jobs = int(os.environ.get("EXP_JOBS", "0"))
    if "--jobs" in args:
        i = args.index("--jobs"); jobs = int(args[i + 1]); del args[i:i + 2]
    if not args or args[0] == "--list":
        print("\n".join(CONFIGS.keys()))
        return
    names = list(CONFIGS.keys()) if args[0] == "--all" else [a for a in args if a in CONFIGS]
    if not names:
        print("no known configs", file=sys.stderr)
        return
    if not jobs:                                   # default: one worker per model
        jobs = len({CONFIGS[n]["model"] for n in names})
    _ensure_eval()
    print(f"# {len(names)} configs over {EVAL_FILE} ({len(EVAL)} items) -> {RESULTS}", flush=True)
    if jobs > 1:
        run_parallel(names, jobs)
    else:
        names.sort(key=lambda n: CONFIGS[n]["model"])
        for name in names:
            try:
                run_config(name)
            except Exception as e:
                import traceback
                print(f"FAILED {name}: {e}", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()

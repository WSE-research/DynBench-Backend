"""
Figures for embedding-optimizations/CONCLUSION.adoc and the cross-method report,
as dependency-free SVG (same engine as rdf2vec-optimizations/_make_charts.py).

    .venv/bin/python experiments/embedding-optimizations/_make_charts.py
"""
import math
from pathlib import Path

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

C1, C2, C3 = "#4C78A8", "#F58518", "#54A24B"     # series palette
CHURT = "#E45756"                                 # "this hurts" / reference
GRID, AXIS, TEXT = "#E6E6E6", "#888888", "#222222"
W = 820


def _ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            out.append(round(v, 10))
        v += step
    return out


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def hbar(path, title, rows, series, colors, xlabel,
         errs=None, vlines=None, value_fmt="{:.3f}", int_commas=False, lmargin=232):
    fmt = (lambda v: f"{v:,.0f}") if int_commas else (lambda v: value_fmt.format(v))
    ns = len(series)
    L, R, T, B = lmargin, 70, 74, 46
    bar_h, grp_gap = 18, 14
    grp_h = ns * bar_h + grp_gap
    ph = len(rows) * grp_h
    H = T + ph + B
    pw = W - L - R
    px0, px1 = L, W - R
    allv = [v for _, vs in rows for v in vs]
    if errs:
        allv += [b for _, es in errs for (_, b) in es] + [a for _, es in errs for (a, _) in es]
    if vlines:
        allv += [x for x, _, _ in vlines]
    xhi = max(allv) * 1.12
    xlo = min(0.0, min(allv) * 1.12)
    if xhi == xlo:
        xhi = xlo + 1

    def X(v):
        return px0 + (v - xlo) / (xhi - xlo) * pw

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W/2}" y="24" text-anchor="middle" font-size="16" '
         f'font-weight="bold" fill="{TEXT}">{_esc(title)}</text>']
    for t in _ticks(xlo, xhi):
        x = X(t)
        s.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" stroke="{GRID}"/>')
        lbl = (f"{t:,.0f}" if int_commas else f"{t:g}")
        s.append(f'<text x="{x:.1f}" y="{T+ph+18}" text-anchor="middle" font-size="11" '
                 f'fill="{AXIS}">{lbl}</text>')
    s.append(f'<text x="{(px0+px1)/2:.1f}" y="{H-8}" text-anchor="middle" font-size="12" '
             f'fill="{TEXT}">{_esc(xlabel)}</text>')
    s.append(f'<line x1="{X(0):.1f}" y1="{T}" x2="{X(0):.1f}" y2="{T+ph}" stroke="{AXIS}"/>')
    for x, lab, col in (vlines or []):
        xx = X(x)
        s.append(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{T+ph}" stroke="{col}" '
                 f'stroke-width="1.5" stroke-dasharray="5,4"/>')
        s.append(f'<text x="{xx:.1f}" y="{T-6}" text-anchor="middle" font-size="10.5" '
                 f'fill="{col}">{_esc(lab)}</text>')
    for i, (label, vs) in enumerate(rows):
        gy = T + i * grp_h
        s.append(f'<text x="{L-10}" y="{gy+grp_h/2+1:.1f}" text-anchor="end" font-size="12" '
                 f'fill="{TEXT}">{_esc(label)}</text>')
        for j, v in enumerate(vs):
            by = gy + j * bar_h + 2
            x0, x1 = (X(0), X(v)) if v >= 0 else (X(v), X(0))
            s.append(f'<rect x="{min(x0,x1):.1f}" y="{by:.1f}" width="{abs(x1-x0):.1f}" '
                     f'height="{bar_h-4}" fill="{colors[j]}"/>')
            cy = by + (bar_h - 4) / 2
            if errs:
                lo, hi = errs[i][1][j]
                s.append(f'<line x1="{X(lo):.1f}" y1="{cy:.1f}" x2="{X(hi):.1f}" y2="{cy:.1f}" '
                         f'stroke="{TEXT}" stroke-width="1.2"/>')
                for ex in (X(lo), X(hi)):
                    s.append(f'<line x1="{ex:.1f}" y1="{cy-3:.1f}" x2="{ex:.1f}" y2="{cy+3:.1f}" '
                             f'stroke="{TEXT}" stroke-width="1.2"/>')
                tx, anc = X(hi) + 5, "start"
            else:
                tx, anc = ((X(v) + 5, "start") if v >= 0 else (X(v) - 5, "end"))
            s.append(f'<text x="{tx:.1f}" y="{cy+4:.1f}" text-anchor="{anc}" font-size="10.5" '
                     f'fill="{TEXT}">{fmt(v)}</text>')
    lx, ly = W - R - 168, T - 36
    if ns > 1 or len(series) > 1:
        for j, name in enumerate(series):
            yy = ly + j * 15
            s.append(f'<rect x="{lx}" y="{yy}" width="11" height="11" fill="{colors[j]}"/>')
            s.append(f'<text x="{lx+16}" y="{yy+10}" font-size="11" fill="{TEXT}">{_esc(name)}</text>')
    s.append("</svg>")
    (OUT / path).write_text("\n".join(s), encoding="utf-8")
    print("wrote", OUT / path)


SER2 = ["n=50", "n=150"]
COL2 = [C1, C2]

# Fig 1 — source benchmarks (shared dataset)
hbar("fig1-benchmarks.svg", "Source benchmarks: records and clean gold pairs",
     [("DynQALD", [1791, 1791]), ("DynRuBQ", [4011, 4009]), ("Total", [5802, 5800])],
     ["Records", "Clean gold pairs"], [C1, C3], "count", int_commas=True, lmargin=120)

# Fig 2 — evaluation-set language mix (shared dataset)
hbar("fig2-eval-languages.svg", "Evaluation sets: items per language",
     [("English (en)", [27, 64]), ("Russian (ru)", [14, 58]), ("German (de)", [6, 12]),
      ("Ukrainian (uk)", [3, 12]), ("French (fr)", [0, 4])],
     SER2, COL2, "items", int_commas=True, lmargin=140)

# Fig 3 — n=150 leaderboard (top 12 of 45)
hbar("fig3-leaderboard.svg", "Embedding leaderboard (n=150): MRR by configuration",
     [("bge-large-en + label", [0.354]), ("mxbai-large + label", [0.340]),
      ("bge-large-en + label+desc", [0.316]), ("e5-large-v2 + label+desc", [0.314]),
      ("mxbai-large + label+desc", [0.314]), ("e5-large + label+desc", [0.304]),
      ("bge-m3 + label+desc", [0.299]), ("distiluse + label+desc", [0.295]),
      ("e5-base + label+desc", [0.287]), ("LaBSE + label+desc", [0.271]),
      ("mMiniLM baseline (production)", [0.260]), ("mMiniLM + label only", [0.220])],
     ["MRR (n=150)"], [C1], "MRR (higher is better)",
     vlines=[(0.182, "random 0.182", CHURT)])

# Fig 4 — text-template x model interaction (the key finding)
hbar("fig4-template-interaction.svg",
     "Text template x model interaction (n=150, MRR)",
     [("bge-large-en", [0.354, 0.316, 0.267]),
      ("mxbai-large", [0.340, 0.314, 0.257]),
      ("multilingual-e5-large", [0.300, 0.304, 0.251]),
      ("mMiniLM (production)", [0.220, 0.260, 0.276])],
     ["label only", "label + description", "description only"], [C1, C2, C3],
     "MRR (higher is better)", lmargin=180)

# Fig 5 — prompts / instructions do not help symmetric ranking (n=150)
hbar("fig5-prompts.svg",
     "Retrieval prompts and instructions do not help (n=150, MRR)",
     [("bge-large-en", [0.316, 0.300]), ("mxbai-large", [0.314, 0.300]),
      ("multilingual-e5-large-instruct", [0.297, 0.290]), ("multilingual-e5-base", [0.287, 0.262])],
     ["plain / symmetric", "with retrieval prompt/instruction"], [C1, CHURT],
     "MRR (higher is better)", lmargin=210)

# Fig 6 — n=50 vs n=150 stability (top configs)
hbar("fig6-stability.svg", "Stability of top configs: n=50 vs n=150 (MRR)",
     [("bge-large-en + label", [0.413, 0.354]), ("mxbai-large + label", [0.371, 0.340]),
      ("e5-large-v2 + label+desc", [0.355, 0.314]), ("bge-large-en + label+desc", [0.357, 0.316]),
      ("e5-large + label+desc", [0.309, 0.304]), ("mMiniLM baseline", [0.252, 0.260])],
     SER2, COL2, "MRR (higher is better)", lmargin=200)

# Fig 7 — paired bootstrap: optimized embedding vs production (n=150)
hbar("fig7-significance.svg",
     "Optimized vs production embedding: ΔMRR with 95% CI (n=150)",
     [("bge-large-en+label vs production", [0.094]),
      ("bge-large-en+label vs e5-large", [0.050]),
      ("bge-large-en+label vs same model, label+desc", [0.038])],
     ["ΔMRR"], [C1], "ΔMRR (positive favors optimized)",
     errs=[("a", [(0.036, 0.153)]), ("b", [(-0.008, 0.108)]), ("c", [(-0.010, 0.088)])],
     lmargin=300)

# Fig 8 — cross-method best-vs-best (for the combined report)
hbar("fig8-crossmethod.svg",
     "Best-vs-best on the gold metric (n=150, MRR)",
     [("RDF2Vec  global_depth1_fullcorpus", [0.365]),
      ("Embedding  bge-large-en + label", [0.354]),
      ("Embedding production (mMiniLM)", [0.260])],
     ["MRR (n=150)"], [C3], "MRR (higher is better)",
     vlines=[(0.182, "random 0.182", CHURT)], lmargin=250)

# Fig 9 — optimizing the embedding closed the gap to a tie (paired bootstrap)
hbar("fig9-crossmethod-delta.svg",
     "RDF2Vec-best minus embedding-best: ΔMRR with 95% CI (n=150)",
     [("vs UNDER-optimized embedding (e5-large, 0.304)", [0.061]),
      ("vs OPTIMIZED embedding (bge-large-en+label, 0.354)", [0.011])],
     ["ΔMRR"], [C1], "ΔMRR (positive = RDF2Vec ahead; CI crossing 0 = tie)",
     errs=[("a", [(0.000, 0.120)]), ("b", [(-0.056, 0.076)])], lmargin=320)

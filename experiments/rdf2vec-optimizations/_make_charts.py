"""
Generate the figures embedded in CONCLUSION.adoc as dependency-free SVG
(horizontal grouped bar charts, with optional error bars and reference lines).
No third-party libraries so it runs in any venv and the output is diffable.

    .venv/bin/python experiments/rdf2vec-optimizations/_make_charts.py

Writes experiments/rdf2vec-optimizations/charts/*.svg
"""
import math
from pathlib import Path

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

C50, C150 = "#4C78A8", "#F58518"          # n=50 / n=150 (and generic series 1 / 2)
CREF = "#E45756"                          # reference line
GRID = "#E6E6E6"
AXIS = "#888888"
TEXT = "#222222"
W = 780                                   # canvas width


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
         errs=None, vlines=None, value_fmt="{:.3f}", int_commas=False):
    """rows: [(label, [v1, v2, ...]), ...]; series: ['n=50', ...]; errs: [(label,[(lo,hi),...])]."""
    fmt = (lambda v: f"{v:,.0f}") if int_commas else (lambda v: value_fmt.format(v))
    ns = len(series)
    L, R, T, B = 220, 70, 74, 46
    bar_h, grp_gap = 20, 16
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
         f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2}" y="24" text-anchor="middle" font-size="16" '
             f'font-weight="bold" fill="{TEXT}">{_esc(title)}</text>')

    # gridlines + x ticks
    for t in _ticks(xlo, xhi):
        x = X(t)
        s.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" stroke="{GRID}"/>')
        lbl = (f"{t:,.0f}" if int_commas else f"{t:g}")
        s.append(f'<text x="{x:.1f}" y="{T+ph+18}" text-anchor="middle" '
                 f'font-size="11" fill="{AXIS}">{lbl}</text>')
    s.append(f'<text x="{(px0+px1)/2:.1f}" y="{H-8}" text-anchor="middle" '
             f'font-size="12" fill="{TEXT}">{_esc(xlabel)}</text>')
    # zero / axis line
    s.append(f'<line x1="{X(0):.1f}" y1="{T}" x2="{X(0):.1f}" y2="{T+ph}" stroke="{AXIS}"/>')

    # reference vertical lines
    for x, lab, col in (vlines or []):
        xx = X(x)
        s.append(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{T+ph}" '
                 f'stroke="{col}" stroke-width="1.5" stroke-dasharray="5,4"/>')
        s.append(f'<text x="{xx:.1f}" y="{T-6}" text-anchor="middle" font-size="10.5" '
                 f'fill="{col}">{_esc(lab)}</text>')

    # bars
    for i, (label, vs) in enumerate(rows):
        gy = T + i * grp_h
        s.append(f'<text x="{L-10}" y="{gy+grp_h/2+1:.1f}" text-anchor="end" '
                 f'font-size="12" fill="{TEXT}">{_esc(label)}</text>')
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
                tx, anc = (X(hi) + 5, "start")
            else:
                tx, anc = ((X(v) + 5, "start") if v >= 0 else (X(v) - 5, "end"))
            s.append(f'<text x="{tx:.1f}" y="{cy+4:.1f}" text-anchor="{anc}" '
                     f'font-size="10.5" fill="{TEXT}">{fmt(v)}</text>')

    # legend (top-right)
    lx, ly = W - R - 150, T - 36
    for j, name in enumerate(series):
        yy = ly + j * 15
        s.append(f'<rect x="{lx}" y="{yy}" width="11" height="11" fill="{colors[j]}"/>')
        s.append(f'<text x="{lx+16}" y="{yy+10}" font-size="11" fill="{TEXT}">{_esc(name)}</text>')

    s.append("</svg>")
    (OUT / path).write_text("\n".join(s), encoding="utf-8")
    print("wrote", OUT / path)


# --------------------------------------------------------------------------- #
SER2 = ["n=50", "n=150"]
COL2 = [C50, C150]

# Fig 1 — source benchmarks (table 3.1)
hbar("fig1-benchmarks.svg", "Source benchmarks: records and clean gold pairs",
     [("DynQALD", [1791, 1791]), ("DynRuBQ", [4011, 4009]), ("Total", [5802, 5800])],
     ["Records", "Clean gold pairs"], [C50, "#54A24B"], "count",
     int_commas=True)

# Fig 2 — evaluation-set language mix (table 3.2)
hbar("fig2-eval-languages.svg", "Evaluation sets: items per language",
     [("English (en)", [27, 64]), ("Russian (ru)", [14, 58]), ("German (de)", [6, 12]),
      ("Ukrainian (uk)", [3, 12]), ("French (fr)", [0, 4])],
     SER2, COL2, "items", int_commas=True)

# Fig 3 — materialized local graph sizes (table 3.3)
hbar("fig3-graph-sizes.svg", "Materialized local graph: size by sample",
     [("Eval nodes requested", [1310, 3921]), ("Distinct subjects", [5289, 7891]),
      ("Depth-1 objects", [24573, 60217]), ("Distinct objects", [88026, 123217]),
      ("Triples", [173790, 242888])],
     SER2, COL2, "count", int_commas=True)

# Fig 4 — leaderboard MRR (table 5.1)
hbar("fig4-leaderboard.svg", "Gold-substitute ranking: MRR by configuration",
     [("global_depth1", [0.432, 0.358]), ("global_depth1_fullcorpus", [0.425, 0.365]),
      ("depth1_walks_200", [0.351, 0.328]), ("depth_1", [0.336, 0.335]),
      ("embedding_reference", [0.252, 0.260]), ("global_walks_100", [0.227, 0.241]),
      ("baseline_per_request", [0.201, 0.200])],
     SER2, COL2, "MRR (higher is better)",
     vlines=[(0.184, "random ≈ 0.184", CREF)])

# Fig 5 — decisive contrasts, paired bootstrap (table 5.2)
hbar("fig5-contrasts.svg", "Decisive contrasts: ΔMRR with 95% bootstrap CI",
     [("global-depth1 vs production", [0.231, 0.165]),
      ("global-depth1 vs embedding", [0.180, 0.105]),
      ("embedding vs production", [0.050, 0.060]),
      ("depth_1 vs production", [0.135, 0.135])],
     SER2, COL2, "ΔMRR (positive favors first method)",
     errs=[("global-depth1 vs production", [(0.130, 0.336), (0.113, 0.221)]),
           ("global-depth1 vs embedding", [(0.095, 0.270), (0.047, 0.165)]),
           ("embedding vs production", [(-0.033, 0.143), (0.008, 0.114)]),
           ("depth_1 vs production", [(0.057, 0.217), (0.091, 0.181)])])

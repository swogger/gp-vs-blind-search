#!/usr/bin/env python3
"""
Render Figure 12: convergence against target K-complexity, both regimes.

Text drawn inside the plot follows the paper: the axis label states that the
plotted K is minimal only for K <= 11; the force-included target is labelled by
its properties (rigid, gradient-free) rather than by an identifier; and the K50
annotations report the pre-specified-levels fit (8.5 / 6.3), which is the value
quoted in the abstract and Sections 5.4, 6.1 and 7.1.

Curves are fitted to the thirteen pre-specified levels only (K = 7..31). The
post-hoc levels K = 3 and 5 are plotted as hollow markers and excluded from the
fit, so the drawn curve and the quoted midpoint are the same estimate.

Style is fixed by the manuscript: Regime A black/circles/solid, Regime B
grey/triangles/dashed. Identity is carried by marker shape and line style as well
as tone, so the figure survives greyscale printing.
"""
import csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                 "results", "synthetic_dense_k"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "figures"))
Z = 1.96

def load(fn):
    rows = []
    for r in csv.DictReader(open(os.path.join(R, fn))):
        rows.append((r["target_id"], int(r["K_construction"]), r["regime"],
                     1 if r["converged"] == "true" else 0))
    return rows

def rates(rows, regime):
    d = defaultdict(lambda: [0, 0])
    for _, k, reg, y in rows:
        if reg == regime:
            d[k][0] += y; d[k][1] += 1
    return {k: (v[0] / v[1], v[1]) for k, v in sorted(d.items())}

def wilson(p, n):
    if n == 0: return (0.0, 0.0)
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def logit_fit(K, y, iters=300):
    X = np.column_stack([np.ones(len(K)), K.astype(float)]); b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = np.clip(p * (1 - p), 1e-9, None)
        z = X @ b + (y - p) / W
        nb = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ z)
        if np.max(np.abs(nb - b)) < 1e-12: b = nb; break
        b = nb
    return b

pre  = load("synthetic_runs.csv")        # K = 7..31, pre-specified
post = load("synthetic_runs_lowk.csv")   # K = 3, 5, post hoc

fig, ax = plt.subplots(figsize=(9, 6), dpi=340)
series = [("A", "Regime A (crossover + mutation)", "black", "o", "-",  ":"),
          ("B", "Regime B (mutation only)",        "0.45",  "^", "--", "-.")]
kk = np.linspace(2.5, 31.5, 400)

for reg, label, col, mk, ls, vls in series:
    rp, rq = rates(pre, reg), rates(post, reg)
    for src, filled in ((rp, True), (rq, False)):
        xs = sorted(src)
        ys = [src[k][0] for k in xs]
        lo = [src[k][0] - wilson(*src[k])[0] for k in xs]
        hi = [wilson(*src[k])[1] - src[k][0] for k in xs]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ecolor=col,
                    markersize=6.5, capsize=3, elinewidth=1.1, linestyle="none",
                    markerfacecolor=(col if filled else "white"),
                    markeredgecolor=col, markeredgewidth=1.3,
                    label=(label if filled else None), zorder=3)
    sub = [r for r in pre if r[2] == reg]          # fit: pre-specified only
    b = logit_fit(np.array([r[1] for r in sub]), np.array([r[3] for r in sub]))
    k50 = -b[0] / b[1]
    ax.plot(kk, 1 / (1 + np.exp(-(b[0] + b[1] * kk))), ls, color=col,
            linewidth=1.8, zorder=2)
    ax.axvline(k50, color=col, linestyle=vls, linewidth=1.0, alpha=0.75, zorder=1)
    ax.annotate(f"$K_{{50}}$ = {k50:.1f}\n({label.split(' (')[0]})",
                xy=(k50, 0.60 if reg == "A" else 0.38),
                xytext=(k50 + 1.4 if reg == "A" else k50 - 3.9,
                        0.63 if reg == "A" else 0.30),
                fontsize=9.5, color=col)

# the force-included rigid, gradient-free reference case
ax.plot([7], [0.70], marker="*", markersize=17, markerfacecolor="white",
        markeredgecolor="black", markeredgewidth=1.3, linestyle="none",
        label="rigid reference case", zorder=4)
ax.annotate("rigid reference case\n$eml(x, eml(eml(x,x),x))$\n$K$ = 7, rate 0.70",
            xy=(7.15, 0.71), xytext=(10.4, 0.855), fontsize=8.8,
            arrowprops=dict(arrowstyle="-", color="black", linewidth=0.9))

ax.plot([], [], "o", markerfacecolor="white", markeredgecolor="black",
        linestyle="none", markersize=6.5, label="post-hoc $K$ = 3, 5 (exhaustive)")

# order the legend: the two series first, then the marker keys
h, l = ax.get_legend_handles_labels()
order = [l.index(x) for x in
         ["Regime A (crossover + mutation)", "Regime B (mutation only)",
          "post-hoc $K$ = 3, 5 (exhaustive)", "rigid reference case"]]
h, l = [h[i] for i in order], [l[i] for i in order]

ax.set_xlabel("Target $K$-complexity (nodes in the constructed EML representation; "
              "minimal for $K \\leq 11$)", fontsize=10.5)
ax.set_ylabel("Convergence rate", fontsize=11)
ax.set_xticks(range(3, 32, 2)); ax.set_xlim(2, 32); ax.set_ylim(-0.045, 1.06)
ax.legend(h, l, fontsize=9, frameon=True, framealpha=1, edgecolor="0.7",
          loc="upper right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=9.5)
ax.grid(axis="y", color="0.9", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig12_convergence_vs_K.{ext}"),
                dpi=400, bbox_inches="tight", facecolor="white")
print("written ->", OUT)

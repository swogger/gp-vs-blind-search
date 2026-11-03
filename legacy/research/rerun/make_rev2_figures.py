"""Figures 7 and 12 for the second revision (reviewer 1, point 3 and point 2).

    python3 research/rerun/make_rev2_figures.py

Fig 7  : the published rerun bars (reference cell, Regime A, medians over completed runs), with the number
         of completed runs printed above each bar, since the medians rest on very different completion rates.
Fig 12 : the published rerun curve, with the run-level Wilson intervals replaced by cluster-adjusted ones.
         Pre-specified levels K = 7-31: Wilson score interval on the effective sample n / DEFF, where
         DEFF = 1 + (m - 1) * ICC (one-way ANOVA ICC of convergence within targets, m = 10 runs per target)
         and the design effect used is the larger of the level's own estimate and the regime-wide pooled
         estimate (targets nested in levels), so that levels with few or no convergent runs, where the
         level's own ICC is undefined or uninformative, are not given run-level intervals.
         Post-hoc levels K = 3, 5: the target sets are complete enumerations, so there is no between-target
         sampling and the run-level Wilson interval (conditional on those targets) is kept.
Outputs go to figures/ with the suffix _rev2; the interval table is printed for Table 11.
"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RR = os.path.join(ROOT, "results_rerun")
OUT = os.path.join(ROOT, "figures")
Z, M_RUNS, DPI = 1.96, 10, 400


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def wilson(p, n):
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- Fig 7
def fig7():
    names = {"1.1": "Exp(x)", "1.2": "Ln(x)", "1.3": "Addition", "1.4": "Exp Comp.", "2.1": "Sin+Cos",
             "3.1": "Sinh(x)", "3.2": "Arctan(x)", "3.3": "x$^x$", "3.4": "Sigmoid",
             "3.5": "Polynomial", "4.1": "Nested I", "4.2": "Nested II"}
    acc = defaultdict(lambda: {"std": [], "eml": []})
    tot = defaultdict(lambda: {"std": 0, "eml": 0})
    for r in csv.DictReader(open(os.path.join(RR, "P2_bench_ref.csv"))):
        if r["regime"] != "A":
            continue
        tot[r["test_id"]][r["algo"]] += 1
        if r["status"] in ("converged", "budget", "plateau"):
            acc[r["test_id"]][r["algo"]].append(int(r["final_nodes"]))
    order = list(names)
    s = [np.median(acc[t]["std"]) for t in order]
    e = [np.median(acc[t]["eml"]) for t in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w/2, s, w, label="Std-GP nodes", color="#6fa8dc", edgecolor="0.3", lw=0.6)
    ax.bar(x + w/2, e, w, label="EML-GP nodes", color="#b03030", edgecolor="0.3", lw=0.6)
    for i, t in enumerate(order):
        for xi, yi, algo in ((x[i] - w/2, s[i], "std"), (x[i] + w/2, e[i], "eml")):
            ax.text(xi, yi * 1.12, str(len(acc[t][algo])), ha="center", va="bottom", fontsize=7.5, color="0.15")
    ax.set_yscale("log"); ax.set_ylim(0.8, 2000)
    ax.set_xticks(x); ax.set_xticklabels([names[t] for t in order], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Median final tree size (nodes, log scale)", fontsize=10)
    ax.legend(fontsize=9, frameon=True, loc="upper left")
    ax.text(0.995, 0.985, "number above bar: completed runs of 30", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color="0.25")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "Fig07_median_tree_size_rev2")
    for i, t in enumerate(order):
        print(f"  Fig 7 {t:4s} Std {s[i]:6.1f} n={len(acc[t]['std'])}/{tot[t]['std']}   "
              f"EML {e[i]:6.1f} n={len(acc[t]['eml'])}/{tot[t]['eml']}")


# ---------------------------------------------------------------- Fig 12
def load(path):
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        by[(int(r["K_construction"]), r["regime"])][r["target_id"]].append(1 if r["converged"] == "true" else 0)
    return by


def icc(groups):
    """One-way ANOVA intraclass correlation; nan when the level has no between-run variation."""
    ys = [np.asarray(v, float) for v in groups]
    n = sum(len(v) for v in ys); g = len(ys); m = n / g; p = np.concatenate(ys).mean()
    msb = sum(len(v) * (v.mean() - p)**2 for v in ys) / (g - 1)
    msw = sum(((v - v.mean())**2).sum() for v in ys) / (n - g)
    den = msb + (m - 1) * msw
    return (msb - msw) / den if den > 0 else float("nan")


def pooled_icc(by, reg):
    ssb = ssw = dfb = dfw = 0.0
    for (k, rg), t in by.items():
        if rg != reg:
            continue
        ys = [np.asarray(v, float) for v in t.values()]; p = np.concatenate(ys).mean()
        ssb += sum(len(v) * (v.mean() - p)**2 for v in ys); dfb += len(ys) - 1
        ssw += sum(((v - v.mean())**2).sum() for v in ys); dfw += sum(len(v) for v in ys) - len(ys)
    msb, msw = ssb / dfb, ssw / dfw
    return (msb - msw) / (msb + (M_RUNS - 1) * msw)


def logit_fit(K, y, iters=300):
    X = np.column_stack([np.ones(len(K)), K.astype(float)]); b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = np.clip(p * (1 - p), 1e-9, None)
        nb = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ (X @ b + (y - p) / W))
        if np.max(np.abs(nb - b)) < 1e-12:
            return nb
        b = nb
    return b


def fig12():
    pre = load(os.path.join(RR, "P1_synth_main.csv"))     # K = 7..31, pre-specified
    post = load(os.path.join(RR, "P1b_synth_lowk.csv"))   # K = 3, 5, complete enumerations
    fig, ax = plt.subplots(figsize=(9, 6), dpi=340)
    series = [("A", "Regime A (crossover + mutation)", "black", "o", "-", ":"),
              ("B", "Regime B (mutation only)", "0.45", "^", "--", "-.")]
    kk = np.linspace(2.5, 31.5, 400)
    print("  Table 11 intervals (rate, cluster-adjusted 95% CI, design effect used):")
    for reg, label, col, mk, ls, vls in series:
        deff_pool = 1 + (M_RUNS - 1) * pooled_icc(pre, reg)
        print(f"   Regime {reg}: pooled ICC design effect {deff_pool:.2f}")
        for src, filled in ((pre, True), (post, False)):
            xs, ys, lo, hi = [], [], [], []
            for k in sorted(kk_ for kk_, rg in src if rg == reg):
                groups = list(src[(k, reg)].values())
                n = sum(len(v) for v in groups); p = sum(sum(v) for v in groups) / n
                if filled:
                    r = icc(groups)
                    own = 1 + (M_RUNS - 1) * r if (r == r and r > 0) else 1.0
                    deff = max(own, deff_pool)
                else:
                    deff = 1.0
                a, b = wilson(p, n / deff)
                xs.append(k); ys.append(p); lo.append(max(0.0, p - a)); hi.append(max(0.0, b - p))
                print(f"    K={k:2d} {reg}  {p:.3f}  [{a:.3f}, {b:.3f}]  DEFF {deff:.2f}"
                      + ("" if filled else "  (run-level Wilson; complete target set)"))
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ecolor=col, markersize=6.5, capsize=3,
                        elinewidth=1.1, linestyle="none", markerfacecolor=(col if filled else "white"),
                        markeredgecolor=col, markeredgewidth=1.3, label=(label if filled else None), zorder=3)
        K = np.array([k for (k, rg), t in pre.items() if rg == reg for v in t.values() for _ in v])
        Y = np.array([y for (k, rg), t in pre.items() if rg == reg for v in t.values() for y in v])
        b = logit_fit(K, Y); k50 = -b[0] / b[1]
        print(f"   Regime {reg}: K50 = {k50:.2f}")
        ax.plot(kk, 1 / (1 + np.exp(-(b[0] + b[1] * kk))), ls, color=col, linewidth=1.8, zorder=2)
        ax.axvline(k50, color=col, linestyle=vls, linewidth=1.0, alpha=0.75, zorder=1)
        ax.annotate(f"$K_{{50}}$ = {k50:.1f}\n({label.split(' (')[0]})",
                    xy=(k50, 0.60 if reg == "A" else 0.38),
                    xytext=(k50 + 1.4 if reg == "A" else k50 - 3.9, 0.63 if reg == "A" else 0.30),
                    fontsize=9.5, color=col)
    ref = [r for r in csv.DictReader(open(os.path.join(RR, "P1_synth_main.csv"))) if r["target_id"] == "reviewer_ce"]
    rate = np.mean([r["converged"] == "true" for r in ref])
    ax.plot([7], [rate], marker="*", markersize=17, markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.3, linestyle="none", label="rigid reference case", zorder=4)
    ax.annotate(f"rigid reference case\n$eml(x, eml(eml(x,x),x))$\n$K$ = 7, rate {rate:.2f}",
                xy=(7.15, rate - 0.01), xytext=(12.5, 0.93), fontsize=8.8,
                arrowprops=dict(arrowstyle="-", color="black", linewidth=0.9))
    ax.plot([], [], "o", markerfacecolor="white", markeredgecolor="black", linestyle="none", markersize=6.5,
            label="post-hoc $K$ = 3, 5 (exhaustive)")
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in ["Regime A (crossover + mutation)", "Regime B (mutation only)",
                                  "post-hoc $K$ = 3, 5 (exhaustive)", "rigid reference case"]]
    ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=9, frameon=True, framealpha=1,
              edgecolor="0.7", loc="upper right")
    ax.set_xlabel("Target $K$-complexity (nodes in the constructed EML representation; "
                  "minimal for $K \\leq 11$)", fontsize=10.5)
    ax.set_ylabel("Convergence rate", fontsize=11)
    ax.set_xticks(range(3, 32, 2)); ax.set_xlim(2, 32); ax.set_ylim(-0.045, 1.06)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9.5)
    ax.grid(axis="y", color="0.9", linewidth=0.7, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "Fig12_convergence_vs_K_rev2")


if __name__ == "__main__":
    with plt.rc_context({"font.family": ["Arial"]}):   # as the published Fig 7 (via plot_convergence.py)
        fig7()
    fig12()

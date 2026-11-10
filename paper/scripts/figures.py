"""The paper's figures from the recorded analysis files (D116; REQ-ANA-100). Nothing is typed in: every plotted
value is read from data/analysis/*.json, the run records and the enumeration counts.

Outputs: Figure_1.pdf to Figure_5.pdf (vector, fonts embedded) and graphical_abstract.{svg,pdf,png} for the
submission; fig_*.svg for the author-facing HTML pages. Run: PYTHONPATH=src .venv/bin/python paper/scripts/figures.py
Figure 4 and the graphical abstract are hand-drawn SVG (coverage_disc.py) converted with rsvg-convert."""
import collections
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, VERM, GRAY, INK, GREEN = "#0072B2", "#D55E00", "#666666", "#222222", "#009E73"
# one palette for every figure (D123): blue = one GP run and its guidance, vermilion = restarts, grey = exact blind
# search, green = blind search read with GP's estimator (and the estimator's offset O), light blue = new larger
# functions (size spread), warm grey = repeats (repetition), ink = the gap; Figure 4 uses the same hues
LIGHT_BLUE, REPEAT = "#99C7E0", "#C8C6BE"
PALETTE = {BLUE, VERM, GRAY, INK, GREEN, LIGHT_BLUE, REPEAT, "#FFFFFF"}
SVG_NAMES = {1: "fig_success.svg", 2: "fig_price.svg", 3: "fig_reach.svg", 4: "fig_coverage_disc.svg", 5: "fig_waterfall.svg"}
E0 = 20000
LINE_PT = 390.0                                  # the manuscript's line width (elsarticle preprint, 12 pt)
SHARE = {1: 1.0, 2: 1.0, 3: 0.95, 4: 1.0, 5: 0.85}   # each figure's printed width as a share of the line width
EPOCH = "1793836800"                             # SOURCE_DATE_EPOCH for rsvg-convert (2026-11-05): no build date


def _size(n, height_in):
    """Figures are drawn at their printed width (REQ-ANA-105), so a font size in points is the printed size."""
    return (SHARE[n] * LINE_PT / 72, height_in)


plt.rcParams.update({
    "svg.hashsalt": "gp-search-mechanics",
    "font.size": 8, "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})


def _neg(s):
    return s.replace("-", "−")


def fig_success(root):
    """Figure 1: exact success against difficulty at E0 for one calibration set per domain: the measured rate per
    level, the logistic fit and its 50 percent point K50, the exact null's step, and the 50 percent point of the null
    read with the same estimator. No labels inside the panels; one legend below names every element (D123)."""
    from reach.analysis.s2 import confirm, h6d, law
    from reach.analysis.s2.load import read_phase, tidy
    fig, axes = plt.subplots(1, 2, figsize=_size(1, 2.85), sharey=True)
    vals = {}
    for ax, s in zip(axes, ("R-ARITH4", "BOOL-AON-n5")):
        rows = [r for ph in ("s2_cal_a0", "s2_cal_a16") for r in read_phase(ph, root)
                if r.get("set") == s and r.get("algorithm") == "A" and r.get("config") == "base"]
        confirm._load_units(root, [s], "calibration")
        recs = [r for r in confirm._with_units(tidy(rows, (1,))) if r["units"].get("bits_sem") is not None]
        by = collections.defaultdict(lambda: [0, 0])
        for r in recs:
            by[round(r["units"]["bits_sem"], 4)][0] += r["y"]
            by[round(r["units"]["bits_sem"], 4)][1] += 1
        xs = sorted(by)
        ys = [by[x][0] / by[x][1] for x in xs]
        f = law.fit1(np.array([r["units"]["bits_sem"] for r in recs], float), np.array([r["y"] for r in recs], float))
        U = {l["K"]: l["U"] for l in json.loads((Path(root) / f"data/enumeration_s2/{s}/counts.json").read_text())["levels"]}
        k_en = h6d.k50_rates([dict(K=r["K"], bits_sem=r["units"]["bits_sem"]) for r in recs], h6d.enum_probs(E0, U))
        grid = np.linspace(min(xs) - 1, max(xs) + 1, 300)
        ax.step([grid[0], math.log2(E0), math.log2(E0), grid[-1]], [1, 1, 0, 0], where="post", color=GRAY, lw=1.4,
                ls=(0, (4, 2)), zorder=2, label="exact blind search")
        ax.plot(grid, 1 / (1 + np.exp(-f.b1 * (grid - f.k50))), color=BLUE, lw=1.6, zorder=3, label="GP: logistic fit")
        ax.plot(xs, ys, "o", color=BLUE, ms=4.2, zorder=4, label="GP: success at a target level")
        ax.plot([f.k50], [0.5], ls="none", marker="D", ms=6.5, color=BLUE, mec="white", mew=0.9, zorder=5,
                label="GP's reach, K₅₀")
        ax.plot([k_en], [0.5], ls="none", marker="D", ms=6.5, color=GREEN, mfc="white", mec=GREEN, mew=1.5, zorder=5,
                label="blind search read with GP's estimator")
        ax.set_title(s + (" (real)" if not s.startswith("BOOL") else " (Boolean)"), fontsize=8.5)
        ax.set_xlabel("difficulty (semantic bits)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="y", color=GRAY, alpha=0.18, lw=0.7)
        ax.set_axisbelow(True)
        vals[s] = dict(k50=f.k50, null_estimator=k_en, offset=k_en - math.log2(E0))
    axes[0].set_ylabel("exact success at E₀")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 1, 3, 0, 4]                                     # GP's three elements, then the two readings of the null
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, handlelength=2.4, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return fig, vals


def fig_price(root):
    """Figure 2: evaluations per extra bit of target difficulty, real calibration sets, against the null's factor 2."""
    cal = json.loads((Path(root) / "data/analysis/calibration.json").read_text())
    rows = []
    for name, e in cal["h6"].items():
        if e["domain"] == "real":
            r = e["bits_per_doubling"]
            rows.append((name, 2 ** (1 / r["rate"]), 2 ** (1 / r["hi"]), 2 ** (1 / r["lo"])))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=_size(2, 2.5))
    for y, (name, f, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.5, solid_capstyle="round", zorder=2)
        ax.plot([f], [y], "o", color=BLUE, ms=5, zorder=3)
        ax.annotate(f"×{f:.1f}", (hi, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.axvline(2, color=GRAY, lw=1.3, ls=(0, (4, 2)), zorder=1)
    ax.set_ylim(-0.6, len(rows) + 0.2)
    ax.annotate("exact blind search: ×2 per bit", (2, len(rows) - 0.25), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=7.5, color=GRAY)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xticklabels(["×2", "×4", "×8", "×16", "×32"])
    ax.set_xlabel("evaluations needed per additional bit of target difficulty (log scale)")
    ax.grid(axis="x", color=GRAY, alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, dict(rows=rows)


def fig_reach(root):
    """Figure 3: reach against budget on the six held-out real sets, one run and restarts, against both null readings."""
    hr = json.loads((Path(root) / "data/analysis/heldout_reach.json").read_text())
    h11 = json.loads((Path(root) / "data/analysis/h11.json").read_text())
    budgets = [E0 * b for b in (1, 2, 4, 8, 16)]
    one = [hr["summary"]["one_run_mean"][str(b // E0)] for b in budgets]
    null = [math.log2(b) for b in budgets]
    null_est = [hr["null_estimator_by_budget"][str(b // E0)]["mean"] for b in budgets]
    Rs = [2, 4, 8, 16, 32]
    restart = [sum(h11["descriptive"][s]["restarts_L100"][str(r)]["k50"] for s in h11["sets"]) / len(h11["sets"]) for r in Rs]
    fig, ax = plt.subplots(figsize=_size(3, 3.2))
    ax.plot(budgets, null, color=GRAY, lw=1.6, ls=(0, (4, 2)), zorder=2, label="exact blind search, exact reach (log₂E)")
    ax.plot(budgets, null_est, color=GREEN, lw=1.4, ls=(0, (1, 2)), zorder=2, label="exact blind search read with GP's estimator")
    ax.plot(budgets, one, color=BLUE, lw=1.8, marker="o", ms=4.5, zorder=3, label="one long run (held-out mean)")
    ax.plot([r * 10000 for r in Rs], restart, color=VERM, lw=1.8, marker="s", ms=4.5, zorder=3, label="restarted GP, 100-generation runs (held-out mean)")
    xg = budgets[4] * 1.14
    ax.plot([xg, xg], [one[4], null[4]], color=INK, lw=1.8, solid_capstyle="butt", zorder=4)
    for yv in (one[4], null[4]):
        ax.plot([xg / 1.025, xg * 1.025], [yv, yv], color=INK, lw=1.8, zorder=4)
    ax.annotate(_neg(f"gap to the exact reach\nat 16×E₀:\none long run {one[4] - null[4]:+.1f} bits\nrestarts {restart[4] - null[4]:+.1f} bits"),
                (xg, one[4] + 0.3 * (null[4] - one[4])), xytext=(-6, 0), textcoords="offset points", fontsize=8, color=INK, va="center", ha="right")
    ax.set_xlim(budgets[0] / 1.15, budgets[4] * 1.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.21), ncol=1, frameon=False, fontsize=7.5, handlelength=2.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(["E₀", "2×E₀", "4×E₀", "8×E₀", "16×E₀"])
    ax.set_xlabel("evaluation budget")
    ax.set_ylabel("reach K₅₀ (semantic bits)")
    ax.grid(axis="y", color=GRAY, alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, dict(one_run=one, null=null, null_estimator=null_est, restarts=restart)


def fig_waterfall(root):
    """Figure 5: the decomposition at E0 per domain, gap = G - S - R + O, registered values."""
    dom = json.loads((Path(root) / "data/analysis/h6d.json").read_text())["domain"]
    proven = json.loads((Path(root) / "data/analysis/preempt.json").read_text())["bool_bounds"]["readings"]["proven_only"]
    fig, ax = plt.subplots(figsize=_size(5, 3.25))

    def waterfall(x0, terms, labels):
        cum = 0.0
        for i, (v, lab) in enumerate(zip(terms, labels)):
            x = x0 + i
            colour = {"G": BLUE, "−S": LIGHT_BLUE, "−R": REPEAT}.get(lab)      # the hues of Figures 1 to 4 (D123)
            if lab == "O":
                ax.bar(x, v, bottom=cum, width=0.62, color="white", edgecolor=GREEN, hatch="///", lw=0.9, zorder=3)
            else:
                ax.bar(x, v, bottom=cum, width=0.62, color=colour, zorder=3)
            top = cum + v
            ax.annotate(_neg(f"{v:+.1f}"), (x, max(cum, top) + 0.5), ha="center", fontsize=7.5)
            ax.annotate(lab, (x, min(cum, top) - 1.0), ha="center", va="top", fontsize=8)
            if i < len(terms) - 1:
                ax.plot([x + 0.31, x + 1 - 0.31], [top, top], color=GRAY, lw=0.8, zorder=2)
            cum = top
        ax.bar(x0 + len(terms), cum, width=0.62, color=INK, zorder=3)
        ax.annotate(_neg(f"gap\n{cum:+.1f}"), (x0 + len(terms), cum + (1.6 if cum >= 0 else -1.6)), ha="center",
                    va="bottom" if cum >= 0 else "top", fontsize=8, fontweight="bold")
        return cum

    ax.axhline(0, color=GRAY, lw=1.0, zorder=1)
    labs = ["G", "−S", "−R", "O"]
    r, b = dom["real"], dom["bool"]
    ax.set_ylim(-9, 26)
    gaps = dict(real=waterfall(0, [r["G"], -r["S"], -r["R"], r["O"]], labs), bool=waterfall(6, [b["G"], -b["S"], -b["R"], b["O"]], labs))
    ax.annotate("real", (2.5, 23.5), ha="center", fontsize=8.5)
    g_pr, gap_pr = _neg(f"{proven['G_mean']:+.1f}"), _neg(f"{proven['gap']['gap']:+.1f}")
    ax.annotate(f"Boolean (registered unit;\nproven-only reading: G {g_pr}, gap {gap_pr})", (8.5, 23.5), ha="center", fontsize=7.5)
    ax.set_xticks([])
    ax.set_ylabel("bits at E₀")
    ax.annotate("O (hatched): the estimator's offset", (5.7, -8.0), fontsize=7.5, color=INK)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig, dict(real=r, bool=b, gaps=gaps)


def _disc():
    p = Path(__file__).with_name("coverage_disc.py")
    spec = importlib.util.spec_from_file_location("coverage_disc", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def svg_to(src, dst, width=None):
    """rsvg-convert (Homebrew librsvg): PDF keeps the text as text with embedded font subsets; PNG at a given width."""
    if shutil.which("rsvg-convert") is None:
        raise RuntimeError("rsvg-convert not found (brew install librsvg)")
    fmt = dst.suffix[1:]
    cmd = ["rsvg-convert", "-f", fmt, "-o", str(dst)] + (["-w", str(width)] if width else []) + [str(src)]
    subprocess.run(cmd, check=True, env=dict(os.environ, SOURCE_DATE_EPOCH=EPOCH))      # no build date (REQ-ANA-105)


def _save(fig, svg, pdf):
    """Fixed ids (svg.hashsalt) and no dates, so a rebuild gives the same bytes (REQ-ANA-105)."""
    fig.savefig(svg, metadata={"Date": None})
    fig.savefig(pdf, metadata={"CreationDate": None})


def build_all(root, pdf_dir, svg_dir):
    """Every figure in both formats; returns the plotted values for the tests and the record."""
    root, pdf_dir, svg_dir = Path(root), Path(pdf_dir), Path(svg_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for n, (name, fn) in enumerate(((("success", fig_success)), ("price", fig_price), ("reach", fig_reach)), start=1):
        fig, vals = fn(root)
        _save(fig, svg_dir / SVG_NAMES[n], pdf_dir / f"Figure_{n}.pdf")
        plt.close(fig)
        out[name] = vals
    fig, vals = fig_waterfall(root)
    _save(fig, svg_dir / SVG_NAMES[5], pdf_dir / "Figure_5.pdf")
    plt.close(fig)
    out["waterfall"] = vals
    disc = _disc()
    svg, d = disc.render(root, "square")
    (svg_dir / SVG_NAMES[4]).write_text(svg)
    svg_to(svg_dir / SVG_NAMES[4], pdf_dir / "Figure_4.pdf")
    wide, _ = disc.render(root, "wide")
    (pdf_dir / "graphical_abstract.svg").write_text(wide)
    svg_to(pdf_dir / "graphical_abstract.svg", pdf_dir / "graphical_abstract.pdf")
    svg_to(pdf_dir / "graphical_abstract.svg", pdf_dir / "graphical_abstract.png", width=2800)
    out["disc"] = dict(small=d["small"], above=d["above"], bars=d["bars"], runs=d["runs"])
    return out


def main():
    root = Path(__file__).resolve().parents[2]
    out = build_all(root, pdf_dir=root / "paper/submission", svg_dir=root / "manuscript")
    print("figures built;", "one-run mean:", [round(v, 2) for v in out["reach"]["one_run"]],
          "| disc small distinct:", [round(v) for v in out["disc"]["small"]])


if __name__ == "__main__":
    sys.exit(main())

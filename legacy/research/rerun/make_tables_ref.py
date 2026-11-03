"""Reference-cell tables from the corrected-fitness rerun (Tables 6, 7, 8, 10 and a supplementary
original-vs-rerun comparison), using the statistics code of research/eml_gp_stats.py.

    python3 research/rerun/make_tables_ref.py > results_rerun/tables_ref.md

Sources: results_rerun/P2_bench_ref.csv (all 12 targets, both algorithms, both regimes, 30 seeds),
results_rerun/P4_ssc_uy.csv (Uy-bounded SSC), results_rerun/P6_native_grid.csv (ln across the grid),
the original main-grid files (results/standard_gp, results/mutation_only) and trajectory files.
"""
import csv, os, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "findings"))
from eml_gp_stats import compare_convergence, compare_tree_size, fmt_p, fmt_effect   # noqa: E402
from instruments import main_grid, traj, TESTS                                     # noqa: E402

RR = os.path.join(ROOT, "results_rerun")
NAMES = {"1.1": "Exp(x)", "1.2": "Ln(x)", "1.3": "Addition", "1.4": "Exp Comp.", "2.1": "Sin+Cos", "3.1": "Sinh(x)",
         "3.2": "Arctan(x)", "3.3": "x^x", "3.4": "Sigmoid", "3.5": "Polynomial", "4.1": "Nested I", "4.2": "Nested II"}
K = {"1.1": "3", "1.2": "7", "1.3": "19", "1.4": "≤53", "2.1": "≤637", "3.1": "≤81", "3.2": "≤355", "3.3": "≤25",
     "3.4": "≤53", "3.5": "≤135", "4.1": "≤411", "4.2": "≤1085"}
BLOAT = ["1.4", "2.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2"]
COMPLETED = ("converged", "budget", "plateau")        # cpu_limit and crash runs return no final tree

P2 = list(csv.DictReader(open(os.path.join(RR, "P2_bench_ref.csv"))))
P4 = list(csv.DictReader(open(os.path.join(RR, "P4_ssc_uy.csv"))))
P6 = list(csv.DictReader(open(os.path.join(RR, "P6_native_grid.csv"))))


def runs(rows, t, algo, reg):
    out = sorted((r for r in rows if r["test_id"] == t and r["algo"] == algo and r["regime"] == reg),
                 key=lambda r: int(r["run"]))
    assert len(out) == 30, (t, algo, reg, len(out))
    return out


def holm(ps):
    idx = [i for i, p in enumerate(ps) if not np.isnan(p)]
    out = [np.nan] * len(ps)
    if idx:
        adj = multipletests([ps[i] for i in idx], method="holm")[1]
        for j, i in enumerate(idx):
            out[i] = adj[j]
    return out


def sci(x):
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.3g}"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(abs(m), 1) >= 10:
        e += 1; m = x / 10 ** e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.1f}×10{sup}"


EXACT = 1e-20   # MSE at or below this (RMS error 1e-10, inside the 1e-9 equivalence criterion) counts as exact


def paired(a, b):
    """Seed-paired comparison. d = b - a (positive when a is lower). Returns p, matched-pairs
    rank-biserial r (positive when a is lower), pairs a lower, pairs b lower.
    Errors at or below EXACT are set to zero, so that round-off differences between exact
    solutions (MSE ~1e-30) are ties rather than differences."""
    a = np.where(np.asarray(a, float) <= EXACT, 0.0, np.asarray(a, float))
    b = np.where(np.asarray(b, float) <= EXACT, 0.0, np.asarray(b, float))
    d = b - a
    nz = d[d != 0]
    if len(nz) == 0:
        return np.nan, np.nan, 0, 0
    rk = rankdata(np.abs(nz))
    r = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    return wilcoxon(nz).pvalue, r, int((d > 0).sum()), int((d < 0).sum())


def table6(reg):
    res = []
    for t in TESTS:
        s = pd.Series([float(r["status"] == "converged") for r in runs(P2, t, "std", reg)])
        e = pd.Series([float(r["status"] == "converged") for r in runs(P2, t, "eml", reg)])
        res.append(compare_convergence(s, e))
    adj = holm([r["p_value"] for r in res])
    print(f"## Table 6 (Regime {reg}): convergence, Std-GP vs EML-GP, reference cell, rerun\n")
    print("| ID | Test Case | Std-GP rate | EML-GP rate | Δ | p (raw) | p (Holm) | Significant |")
    print("|---|---|---|---|---|---|---|---|")
    for t, r, h in zip(TESTS, res, adj):
        print(f"| {t} | {NAMES[t]} | {r['std_rate']:.2f} | {r['eml_rate']:.2f} | {r['delta']:+.2f} | {fmt_p(r['p_value'])} | {fmt_p(h)} | {'Yes' if h < 0.05 else 'No'} |")
    print(f"\nSignificant: {sum(h < 0.05 for h in adj)} of 12\n")


def table7(reg):
    res = []
    for t in TESTS:
        s = pd.Series([float(r["final_nodes"]) if r["status"] in COMPLETED else np.nan for r in runs(P2, t, "std", reg)])
        e = pd.Series([float(r["final_nodes"]) if r["status"] in COMPLETED else np.nan for r in runs(P2, t, "eml", reg)])
        res.append(compare_tree_size(s, e))
    adj = holm([r["p_value"] for r in res])
    print(f"## Table 7 (Regime {reg}): final tree size over completed runs, reference cell, rerun\n")
    print("| ID | Test | n (Std/EML) | Std-GP median | EML-GP median | p (raw) | p (Holm) | Effect size | Significant |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t, r, h in zip(TESTS, res, adj):
        ins = r.get("insufficient")
        pr = "insufficient data" if ins else fmt_p(r["p_value"])
        print(f"| {t} | {NAMES[t]} | {r['n_std']}/{r['n_eml']} | {r['std_median']:.1f} | "
              f"{'–' if np.isnan(r['eml_median']) else f'{r[chr(101)+chr(109)+chr(108)+chr(95)+chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]:.1f}'} | "
              f"{pr} | {'–' if ins else fmt_p(h)} | {'–' if ins else fmt_effect(r['effect_size'])} | {'–' if ins else ('Yes' if h < 0.05 else 'No')} |")
    comp = {a: np.mean([r["status"] in COMPLETED for t in BLOAT for r in runs(P2, t, a, reg)]) for a in ("eml", "std")}
    print(f"\nCompleted share on the eight bloat-prone targets: EML-GP {comp['eml']:.0%}, Std-GP {comp['std']:.0%}; "
          f"testable {sum(not r.get('insufficient') for r in res)}, significant {sum(h < 0.05 for h in adj if not np.isnan(h))}\n")


def table10():
    rows = []
    for t in TESTS:
        a = runs(P2, t, "eml", "A"); b = runs(P2, t, "eml", "B")
        ea = [float(r["final_err"]) for r in a]; eb = [float(r["final_err"]) for r in b]
        p, r, a_low, b_low = paired(ea, eb)
        cens = (sum(r["status"] == "cpu_limit" for r in a), sum(r["status"] == "cpu_limit" for r in b))
        rows.append([t, np.median(ea), np.median(eb), p, r, a_low, b_low, cens])
    adj = holm([x[3] for x in rows])
    print("## Table 10: EML-GP Regime A vs Regime B, reference cell, rerun (d = B − A; r > 0 means A lower)\n")
    print("| ID | Test | K | Regime A | Regime B | Pairs A lower / B lower | r | p (raw) | p (Holm) | Direction | CPU-limited A/B |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (t, ma, mb, p, r, al, bl, c), h in zip(rows, adj):
        if np.isnan(p):
            direction, rs, ps, hs = "identical", "–", "–", "–"
        else:
            direction = ("A better" if r > 0 else "B better") if h < 0.05 else "no significant difference"
            rs, ps, hs = f"{r:+.2f}".replace("-", "−"), fmt_p(p), fmt_p(h)
        print(f"| {t} | {NAMES[t]} | {K[t]} | {sci(ma)} | {sci(mb)} | {al} / {bl} | {rs} | {ps} | {hs} | {direction} | {c[0]}/{c[1]} |")
    print()
    return rows, adj


def stability():
    """Direction and uncorrected significance of the three established regime effects in every grid cell."""
    print("## Table 10 stability across the grid (paired Wilcoxon, uncorrected)\n")
    cells = [(g, p) for g in (100, 200, 500) for p in (100, 200, 500)]
    for t in ("1.2", "1.3", "3.1"):
        out = []
        for g, p in cells:
            if t == "1.2":
                src = P2 if (g, p) == (200, 100) else [r for r in P6 if r["G"] == str(g) and r["P"] == str(p)]
                ea = [float(r["final_err"]) for r in sorted((r for r in src if r["test_id"] == t and r["algo"] == "eml" and r["regime"] == "A"), key=lambda r: int(r["run"]))]
                eb = [float(r["final_err"]) for r in sorted((r for r in src if r["test_id"] == t and r["algo"] == "eml" and r["regime"] == "B"), key=lambda r: int(r["run"]))]
            else:   # bit-identical to the original: use the trajectory files (6-decimal log); a few are missing
                ea, eb = [], []
                for r in range(1, 31):
                    try:
                        a_, b_ = traj("A", "eml", t, g, p, r)[-1][1], traj("B", "eml", t, g, p, r)[-1][1]
                    except FileNotFoundError:
                        continue
                    ea.append(a_); eb.append(b_)
            pv, r, al, bl = paired(ea, eb)
            out.append((g, p, pv, r, len(ea)))
        sig = [x for x in out if not np.isnan(x[2]) and x[2] < 0.05]
        short = [f"G{x[0]}P{x[1]} n={x[4]}" for x in out if x[4] < 30]
        same = [x for x in out if not np.isnan(x[3]) and (x[3] > 0) == (t == "1.2")]
        print(f"- {NAMES[t]}: expected direction in {len(same)}/9 cells; significant (p < 0.05) in that direction in "
              f"{sum((x[3] > 0) == (t == '1.2') for x in sig)}/9 cells; " +
              ", ".join(f"G{g}P{p}: r={r:+.2f} p={pv:.3g}" for g, p, pv, r, n in out) + (f" (incomplete pairs: {short})" if short else ""))
    print()


def table8():
    print("## Table 8: SSC (Uy bounds) vs standard crossover, reference cell, corrected fitness\n")
    ssc_tests = ["1.1", "1.2", "1.3", "2.1", "3.2", "3.3"]
    stats_ = {}
    for algo in ("std", "eml"):
        res = []
        for t in ssc_tests:
            x = runs(P2, t, algo, "A")
            s = sorted((r for r in P4 if r["test_id"] == t and r["algo"] == algo), key=lambda r: int(r["run"]))
            assert [r["run"] for r in s] == [r["run"] for r in x] or [int(r["run"]) for r in s] == [int(r["run"]) for r in x]
            ex = [float(r["final_err"]) for r in x]; es = [float(r["final_err"]) for r in s]
            p, rr, x_low, s_low = paired(ex, es)          # rr > 0: standard crossover lower
            res.append((t, np.median(ex), np.median(es), p, rr, x_low, s_low,
                        sum(r["status"] == "cpu_limit" for r in x), sum(r["status"] == "cpu_limit" for r in s)))
        adj = holm([z[3] for z in res])
        stats_[algo] = (res, adj)
    mut = {t: np.median([float(r["final_err"]) for r in runs(P2, t, "eml", "B")]) for t in ssc_tests}
    print("| ID | Test | Std-GP crossover | Std-GP SSC | EML-GP crossover | EML-GP SSC | EML-GP mutation only | SSC effect on EML-GP | SSC effect on Std-GP |")
    print("|---|---|---|---|---|---|---|---|---|")

    def eff(z, h):
        if np.isnan(z[3]):
            return "no change"
        lab = "no significant difference" if h >= 0.05 else ("worse" if z[4] > 0 else "better")
        return f"{lab} (Holm p = {h:.2f}; SSC lower in {z[6]} of 30)"
    for i, t in enumerate(ssc_tests):
        s, sh = stats_["std"][0][i], stats_["std"][1][i]
        e, eh = stats_["eml"][0][i], stats_["eml"][1][i]
        print(f"| {t} | {NAMES[t]} | {sci(s[1])} | {sci(s[2])} | {sci(e[1])} | {sci(e[2])} | {sci(mut[t])} | {eff(e, eh)} | {eff(s, sh)} |")
    print("\nCPU-limited runs (crossover/SSC): " + "; ".join(
        f"{a} {z[0]}: {z[7]}/{z[8]}" for a in ("eml", "std") for z in stats_[a][0] if z[7] or z[8]) + "\n")


def comparison_table():
    print("## New supplementary table: original configuration vs rerun at the reference cell (convergence, of 30; censored)\n")
    print("| ID | Test | Algo | Regime | Converged original | Converged rerun | Censored original (wall clock) | Censored rerun (CPU) |")
    print("|---|---|---|---|---|---|---|---|")
    for t in TESTS:
        for algo in ("std", "eml"):
            for reg in "AB":
                mg = main_grid(reg)[(algo, 200, 100, t)]
                rr = runs(P2, t, algo, reg)
                o = sum(s[0] == "CONVERGED" for s in mg); n = sum(r["status"] == "converged" for r in rr)
                oc = sum(s[0] == "TIMEOUT" for s in mg); nc = sum(r["status"] == "cpu_limit" for r in rr)
                if o != n or oc != nc:
                    print(f"| {t} | {NAMES[t]} | {'Std-GP' if algo == 'std' else 'EML-GP'} | {reg} | {o} | {n} | {oc} | {nc} |")
    print("\n(rows omitted where both columns agree)\n")


def censoring_summary():
    print("## Reference-cell censoring in the rerun (bloat-prone targets, CPU limit)\n")
    for reg in "AB":
        for algo in ("eml", "std"):
            n = sum(r["status"] == "cpu_limit" for t in BLOAT for r in runs(P2, t, algo, reg))
            print(f"- Regime {reg}, {algo}: {n} of 240")
    crash = [r["key"] for r in P2 if r["status"] == "crash"]
    print(f"- crashes: {crash}\n")


if __name__ == "__main__":
    table6("A"); table6("B")
    table7("A")
    table10(); stability()
    table8()
    comparison_table()
    censoring_summary()

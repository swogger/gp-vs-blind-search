"""Manuscript tables for the synthetic dense-K experiment (Tables 11-14, S7, S8), from the rerun.

    python3 research/rerun/make_tables_p1.py > results_rerun/tables_P1.md

Every value printed here is computed from results_rerun/P1_synth_main.csv and P1b_synth_lowk.csv;
nothing is typed in by hand.
"""
import csv, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "synthetic"))
import analyze_p1 as P            # noqa: E402  (K50 fit and data paths)
import analysis_ncrit_vs_k as M   # noqa: E402  (Table 12 model code)

SYN = P.SYN
EXTRA = {t["target_id"]: t for t in csv.DictReader(open(os.path.join(SYN, "synthetic_targets_extra.csv")))}
LOWK_TG = {t["target_id"]: t for t in csv.DictReader(open(os.path.join(SYN, "synthetic_targets_lowk.csv")))}
MINUS = "−"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sci(x, digits=1):
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e3:
        return f"{x:.{digits + 2}g}"
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    if round(abs(m), digits) >= 10:          # 9.96e-7 -> 1.0e-6, not 10.0e-7
        e += 1; m = x / 10 ** e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.{digits}f} × 10{sup}"


def pfmt(p):
    return "< 0.001" if p < 1e-3 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def conv(r):
    return r["converged"] == "true"


def exact(r):
    return conv(r) and r["exact_hash_match"] == "true"


def table11(main, low):
    print("## Table 11 (replace body)\n")
    print("| K | Targets | Rigid | Regime A conv. | 95% CI | A exact | Regime B conv. | 95% CI | B exact | Med. gen (A) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for runs, post in ((low, True), (main, False)):
        for k in sorted({int(r["K_construction"]) for r in runs}):
            rk = [r for r in runs if int(r["K_construction"]) == k]
            tids = {r["target_id"] for r in rk}
            rigid = "–" if post else str(sum(P.TG[t]["rigid_subset"] == "true" for t in tids))
            cells = []
            for reg in "AB":
                rr = [r for r in rk if r["regime"] == reg]
                c, e = sum(map(conv, rr)), sum(map(exact, rr))
                lo, hi = wilson(c, len(rr))
                cells += [f"{c/len(rr):.3f}", f"{lo:.3f}–{hi:.3f}", f"{e/len(rr):.3f}"]
            gens = [float(r["gens_to_converge"]) for r in rk if r["regime"] == "A" and conv(r)]
            mg = "–" if post or not gens else f"{np.median(gens):g}"
            print(f"| {k}{'*' if post else ''} | {len(tids)} | {rigid} | " + " | ".join(cells) + f" | {mg} |")
    print()


def design_effect(main, reg="A", ks=range(7, 18, 2)):
    out = {}
    for k in ks:
        by = defaultdict(list)
        for r in main:
            if int(r["K_construction"]) == k and r["regime"] == reg:
                by[r["target_id"]].append(float(conv(r)))
        g = list(by.values()); m = np.mean([len(x) for x in g]); allv = np.concatenate(g); N = len(allv); a = len(g)
        msb = sum(len(x) * (np.mean(x) - allv.mean()) ** 2 for x in g) / (a - 1)
        msw = sum(((np.array(x) - np.mean(x)) ** 2).sum() for x in g) / (N - a)
        icc = max(0.0, (msb - msw) / (msb + (m - 1) * msw)) if msb + (m - 1) * msw > 0 else 0.0
        out[k] = 1 + (m - 1) * icc
    return out


def table12(main):
    print("## Table 12 (replace body; pooled over both regimes, cluster-robust p by target)\n")
    rows = []
    for r in main:
        t = P.TG[r["target_id"]]
        rows.append({"y": float(conv(r)), "K": float(r["K_construction"]), "alpha": float(t["alpha_rigidity"]),
                     "n_crit": float(t["n_crit"]), "g": float(t["gradient_absence_g"] == "true"), "tid": r["target_id"]})
    groups = [r["tid"] for r in rows]
    print("| Model | Deviance | AUC | Key coefficient |")
    print("|---|---|---|---|")
    for cols, key in ((["K"], "K"), (["n_crit"], "n_crit"), (["K", "alpha"], "alpha"), (["K", "alpha", "g"], "g")):
        f = M.fit(rows, cols, groups)
        v = f["params"][key]; p = f["cluster_p"][key]
        coef = f"{key.replace('alpha', 'α')}: {v:+.2f}".replace("-", MINUS)
        if key in ("alpha", "g"):
            coef += f" (p = {p:.3f})" if p < 0.01 else f" (p = {p:.2f})"
        print(f"| convergence ~ {' + '.join(cols).replace('alpha', 'α')} | {f['deviance']:.1f} | {f['auc']:.3f} | {coef} |")
    print()
    return rows, groups


def s7(main):
    print("## Table S7 (replace body)\n")
    print("| Rigidity rule | conv ~ n_crit AUC | α coefficient in conv ~ K + α | p (cluster-robust) |")
    print("|---|---|---|---|")
    base = None
    for label, col in (("Median over local edits (primary)", "alpha_rigidity_median_rule"),
                       ("Minimum over local edits", "alpha_rigidity_min_rule"),
                       ("Child-swap only", "alpha_rigidity_swap_rule")):
        rows = []
        for r in main:
            a = float(EXTRA[r["target_id"]][col]); m = float(P.TG[r["target_id"]]["m_operators"])
            rows.append({"y": float(conv(r)), "K": float(r["K_construction"]), "alpha": a, "n_crit": a * m})
        g = [r["target_id"] for r in main]
        fn = M.fit(rows, ["n_crit"], g); fa = M.fit(rows, ["K", "alpha"], g)
        base = M.fit(rows, ["K"], g)
        print(f"| {label} | {fn['auc']:.3f} | {fa['params']['alpha']:+.2f} | {fa['cluster_p']['alpha']:.3f} |".replace("| -", f"| {MINUS}").replace(" -", f" {MINUS}"))
    print(f"| conv ~ K baseline | {base['auc']:.3f} | – | – |\n")


def table13(main):
    print("## Table 13 (replace body; Holm across the thirteen pre-specified levels)\n")
    by = defaultdict(dict)
    for r in main:
        by[(r["target_id"], r["seed"], int(r["K_construction"]))][r["regime"]] = conv(r)
    rows = []
    for k in sorted({x[2] for x in by}):
        a = sum(v["A"] and not v["B"] for kk, v in by.items() if kk[2] == k)
        b = sum(v["B"] and not v["A"] for kk, v in by.items() if kk[2] == k)
        rows.append([k, a, b, binomtest(a, a + b, 0.5).pvalue if a + b else 1.0])
    order = sorted(range(len(rows)), key=lambda i: rows[i][3]); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(rows) - rank) * rows[i][3])); rows[i].append(run)
    print("| K | A only | B only | McNemar p | Holm p |")
    print("|---|---|---|---|---|")
    for k, a, b, p, h in rows:
        print(f"| {k} | {a} | {b} | {sci(p) if p < 0.001 else pfmt(p) if a + b else chr(0x2013)} | {pfmt(h) if a + b else chr(0x2013)} |")
    A, B = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"| Pooled | {A} | {B} | {sci(binomtest(A, A + B, 0.5).pvalue)} | – |\n")


def table14(main):
    print("## Table 14 (replace body)\n")
    by = defaultdict(dict)
    for r in main:
        # an exact recovery (output-hash match, validated as dense-grid equivalent) counts as zero error, so that
        # round-off differences between exact solutions are ties (nMSE of an exact tree can reach ~1e-8 on
        # low-variance targets)
        exact_ = r["converged"] == "true" and r["exact_hash_match"] == "true"
        by[(r["target_id"], r["seed"], int(r["K_construction"]))][r["regime"]] = 0.0 if exact_ else float(r["final_nmse"])
    print("| Band | Median nMSE (A) | Median nMSE (B) | Pairs A lower / B lower / tied | Wilcoxon p | Direction |")
    print("|---|---|---|---|---|---|")
    for lo, hi in ((7, 11), (13, 17), (19, 31)):
        a, b = np.array([(v["A"], v["B"]) for k, v in by.items() if lo <= k[2] <= hi]).T
        d = a - b; p = wilcoxon(d[d != 0]).pvalue
        na, nb, nt = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
        direction = ("A better" if na > nb else "B better") if p < 0.05 else "no difference"
        print(f"| K = {lo}–{hi} | {sci(float(np.median(a)))} | {sci(float(np.median(b)))} | {na} / {nb} / {nt} | {sci(p) if p < 0.001 else pfmt(p)} | {direction} |")
    print()


def s8(main):
    print("## Table S8 (replace body)\n")
    loose = [r for r in main if conv(r) and r["exact_hash_match"] != "true"]
    print("| K | Regime | n | Median nMSE | Range |")
    print("|---|---|---|---|---|")
    for k in sorted({int(r["K_construction"]) for r in loose}):
        for reg in "AB":
            v = sorted(float(r["final_nmse"]) for r in loose if int(r["K_construction"]) == k and r["regime"] == reg)
            if v:
                rng = "–" if len(v) == 1 else f"{sci(v[0])} – {sci(v[-1])}"
                print(f"| {k} | {reg} | {len(v)} | {sci(float(np.median(v)))} | {rng} |")
    na = sum(r["regime"] == "A" for r in loose)
    low = sum(int(r["K_construction"]) <= 13 for r in loose)
    print(f"| Total | | {len(loose)} | | |\n")
    print(f"Loose: {len(loose)} (A {na}, B {len(loose) - na}); at K <= 13: {low}\n")


def main():
    main_runs, low_runs = P.read(P.NEW[0]), P.read(P.NEW[1])
    table11(main_runs, low_runs)
    de = design_effect(main_runs)
    print("Design effect (Kish, one-way ANOVA ICC, Regime A): " + ", ".join(f"K={k}: {v:.1f}" for k, v in de.items()) + "\n")
    for k in (7, 11):
        rr = [r for r in main_runs if int(r["K_construction"]) == k and r["regime"] == "A"]
        c, n = sum(map(conv, rr)), len(rr); ne = n / de[k]
        lo, hi = wilson(c, n); lo2, hi2 = wilson(c / n * ne, ne)
        print(f"K={k}: naive [{lo:.3f}, {hi:.3f}] -> design-effect adjusted [{lo2:.3f}, {hi2:.3f}] (n_eff {ne:.0f})")
    print()
    table12(main_runs)
    table13(main_runs)
    table14(main_runs)
    s7(main_runs)
    s8(main_runs)
    for metric in ("conv", "exact"):
        for reg in "AB":
            P.RNG = np.random.default_rng(20260926)          # same resampling draw for every estimate
            e, lo, hi = P.k50(main_runs, reg, metric)
            print(f"K50 {metric} Regime {reg}: {e:.1f} [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()

"""Fixed-primitive-set control (Section 5.5): Tables 15-16 and Supplementary S9-S11, from the rerun.

    python3 research/rerun/make_tables_control.py [--partial] > results_rerun/tables_control.md

Arms, all at the reference cell, corrected fitness, seed-matched (runs 1-30):
  tailored  per-target Table 2 sets        results_rerun/P2_bench_ref.csv (algo std)
  ABL6      {+, -, *, /, exp, log}         results_rerun/P3_control_abl6.csv
  FIX8      {+, -, *, /, exp, log, sin, cos} results_rerun/P8_control_fix8.csv
  EML-GP                                   results_rerun/P2_bench_ref.csv (algo eml)
Paired tests: exact McNemar on convergence (discordant seed pairs) and Wilcoxon signed-rank on final
error, Holm-corrected across the testable targets within each comparison and regime.
"""
import csv, os, sys
import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tables_ref import NAMES, TESTS, holm, paired, sci, RR   # noqa: E402

PARTIAL = "--partial" in sys.argv
rd = lambda f: list(csv.DictReader(open(os.path.join(RR, f))))
P2, P3, P8 = rd("P2_bench_ref.csv"), rd("P3_control_abl6.csv"), rd("P8_control_fix8.csv")
ARMS = {"tailored": (P2, "std"), "ABL6": (P3, "std"), "FIX8": (P8, "std"), "EML": (P2, "eml")}
CHANGE = {"1.1": "+ sin, cos", "1.2": "+ sin, cos", "1.3": "+ exp, log, sin, cos", "1.4": "+ sin, cos",
          "2.1": "− tan; + exp, log", "3.1": "− sinh, cosh, tanh; + sin, cos", "3.2": "− tan, arctan",
          "3.3": "− sqrt; + sin, cos", "3.4": "+ sin, cos", "3.5": "+ exp, log, sin, cos", "4.1": "− tan", "4.2": "− tan"}


def runs(arm, t, reg):
    rows, algo = ARMS[arm]
    out = {int(r["run"]): r for r in rows if r["test_id"] == t and r["regime"] == reg and r["algo"] == algo}
    if len(out) != 30:
        if PARTIAL:
            return None
        raise SystemExit(f"incomplete: {arm} {t} {reg} has {len(out)} runs")
    return [out[i] for i in range(1, 31)]


def rate(arm, t, reg):
    r = runs(arm, t, reg)
    return None if r is None else np.mean([x["status"] == "converged" for x in r])


def f2(x):
    return "…" if x is None else f"{x:.2f}"


def table15():
    print("## Table 15: convergence by operator set, both regimes (reference cell, rerun)\n")
    print("| ID | Test | Set change (8-primitive vs Table 2) | 8-prim A | 8-prim B | 6-prim A | 6-prim B | Table 2 set A | Table 2 set B | EML-GP A | EML-GP B |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    means = {}
    for t in TESTS:
        cells = []
        for arm in ("FIX8", "ABL6", "tailored", "EML"):
            for reg in "AB":
                v = rate(arm, t, reg); cells.append(f2(v)); means.setdefault((arm, reg), []).append(v)
        print(f"| {t} | {NAMES[t]} | {CHANGE[t]} | " + " | ".join(cells) + " |")
    m = {k: (None if any(x is None for x in v) else float(np.mean(v))) for k, v in means.items()}
    print("| | Mean | | " + " | ".join(("…" if m[(a, r)] is None else f"{m[(a, r)]:.3f}") for a in ("FIX8", "ABL6", "tailored", "EML") for r in "AB") + " |\n")
    for reg in "AB":
        gap = None if None in (m[("tailored", reg)], m[("EML", reg)]) else m[("tailored", reg)] - m[("EML", reg)]
        for arm in ("FIX8", "ABL6"):
            if gap and m[(arm, reg)] is not None:
                print(f"- Regime {reg}: tailored − EML gap {gap:.3f}; share removed by the uniform {arm} set "
                      f"{(m[('tailored', reg)] - m[(arm, reg)]) / gap:.2f} (remaining Std-GP advantage {m[(arm, reg)] - m[('EML', reg)]:.3f})")
    print()


def compare(a, b, reg):
    """a vs b, seed-matched: McNemar on convergence, Wilcoxon on final error. Returns per-target rows."""
    out = []
    for t in TESTS:
        ra, rb = runs(a, t, reg), runs(b, t, reg)
        if ra is None or rb is None:
            out.append((t, None)); continue
        ca = [x["status"] == "converged" for x in ra]; cb = [x["status"] == "converged" for x in rb]
        a_only = sum(x and not y for x, y in zip(ca, cb)); b_only = sum(y and not x for x, y in zip(ca, cb))
        pm = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else np.nan
        ea = [float(x["final_err"]) for x in ra]; eb = [float(x["final_err"]) for x in rb]
        pw, rr, a_low, b_low = paired(ea, eb)
        out.append((t, dict(rate_a=np.mean(ca), rate_b=np.mean(cb), a_only=a_only, b_only=b_only, pm=pm,
                            pw=pw, r=rr, a_low=a_low, b_low=b_low,
                            cens_a=sum(x["status"] == "cpu_limit" for x in ra), cens_b=sum(x["status"] == "cpu_limit" for x in rb))))
    hm = holm([np.nan if d is None else d["pm"] for _, d in out])
    hw = holm([np.nan if d is None else d["pw"] for _, d in out])
    return out, hm, hw


def table16():
    print("## Table 16: decomposition, seed-matched tests (McNemar on convergence; Wilcoxon on final error; Holm within column)\n")
    for label, a, b in (("Replacing the Table 2 set by the 6-primitive set", "tailored", "ABL6"),
                        ("Adding sin and cos to the 6-primitive set", "ABL6", "FIX8")):
        for reg in "AB":
            out, hm, hw = compare(a, b, reg)
            print(f"### {label}, Regime {reg} ({a} → {b})\n")
            print(f"| ID | Test | conv {a} → {b} | only {a} / only {b} | McNemar Holm p | error: {a} lower / {b} lower | Wilcoxon Holm p | CPU-limited {a}/{b} |")
            print("|---|---|---|---|---|---|---|---|")
            for (t, d), h1, h2 in zip(out, hm, hw):
                if d is None:
                    print(f"| {t} | {NAMES[t]} | … | | | | | |"); continue
                s1 = "–" if np.isnan(h1) else (f"**{h1:.3f}**" if h1 < 0.05 else f"{h1:.2f}")
                s2 = "–" if np.isnan(h2) else (f"**{h2:.3f}**" if h2 < 0.05 else f"{h2:.2f}")
                print(f"| {t} | {NAMES[t]} | {d['rate_a']:.2f} → {d['rate_b']:.2f} | {d['a_only']} / {d['b_only']} | {s1} | "
                      f"{d['a_low']} / {d['b_low']} | {s2} | {d['cens_a']}/{d['cens_b']} |")
            print()


def supp_detail(arm, title):
    print(f"## {title}\n")
    print("| Test | Regime | Conv. | Median error (completed runs) | Median nodes (completed) | CPU-limited | Table 2 set conv. |")
    print("|---|---|---|---|---|---|---|")
    for t in TESTS:
        for reg in "AB":
            r = runs(arm, t, reg)
            if r is None:
                print(f"| {NAMES[t]} | {reg} | … | | | | |"); continue
            comp = [x for x in r if x["status"] in ("converged", "budget", "plateau")]
            me = sci(float(np.median([float(x["final_err"]) for x in comp]))) if comp else "–"
            mn = f"{np.median([int(x['final_nodes']) for x in comp]):g}" if comp else "–"
            print(f"| {NAMES[t]} | {reg} | {np.mean([x['status'] == 'converged' for x in r]):.2f} | {me} | {mn} | "
                  f"{sum(x['status'] == 'cpu_limit' for x in r)} | {f2(rate('tailored', t, reg))} |")
    print()


if __name__ == "__main__":
    table15()
    table16()
    supp_detail("FIX8", "Table S9: 8-primitive control, per target")
    supp_detail("ABL6", "Table S11: 6-primitive control, per target")


def table16_compact():
    """Main-text Table 16: Table 2 set -> 6-primitive -> 8-primitive, per regime, with significance marks."""
    steps = (("tailored", "ABL6"), ("ABL6", "FIX8"))
    res = {(a, b, reg): compare(a, b, reg) for a, b in steps for reg in "AB"}
    print("## Table 16 (compact): Table 2 set → 6 primitives → 8 primitives (c: convergence change significant, "
          "McNemar; e: final-error change significant, Wilcoxon; Holm within step and regime)\n")
    print("| ID | Test | Set change (8-primitive vs Table 2) | Regime A | Regime B |")
    print("|---|---|---|---|---|")
    for i, t in enumerate(TESTS):
        cells = []
        for reg in "AB":
            parts = [f"{rate('tailored', t, reg):.2f}"]
            for a, b in steps:
                out, hm, hw = res[(a, b, reg)]
                d = out[i][1]
                mark = "ᶜ" if not np.isnan(hm[i]) and hm[i] < 0.05 else ("ᵉ" if not np.isnan(hw[i]) and hw[i] < 0.05 else "")
                parts.append(f"{d['rate_b']:.2f}{mark}")
            cells.append(" → ".join(parts))
        print(f"| {t} | {NAMES[t]} | {CHANGE[t]} | {cells[0]} | {cells[1]} |")
    print()


if __name__ == "__main__" and "--compact" in sys.argv:
    table16_compact()

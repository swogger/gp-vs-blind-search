"""Tables S12 and S13 (sensitivity of assembly reach to the generation budget), from the rerun.

    python3 research/rerun/make_tables_p7.py > results_rerun/tables_P7.md

Like-for-like design: the 200-generation row uses the same targets and the same three seeds as the
800-generation sweep (P7), taken from the main rerun (P1, P1b). P7b (3,200 generations) is added when present.
"""
import csv, os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_p1 as P    # noqa: E402  (same logistic fit and target-resampling CI as K50 in Section 5.4)

RR = os.path.join(P.ROOT, "results_rerun")
read = lambda f: list(csv.DictReader(open(os.path.join(RR, f)))) if os.path.exists(os.path.join(RR, f)) else []
P7 = read("P7_synth_budget800.csv")
P7b = read("P7b_synth_budget3200.csv")
keys = {(r["target_id"], r["regime"], r["seed"]) for r in P7}
base = [r for r in read("P1_synth_main.csv") + read("P1b_synth_lowk.csv") if (r["target_id"], r["regime"], r["seed"]) in keys]
CENSORED = ("timeout", "cpu_limit")


def k50(rows, reg, complete_case):
    rr = [r for r in rows if r["regime"] == reg and not (complete_case and r["status"] in CENSORED)]
    P.RNG = np.random.default_rng(20260926)
    return P.k50(rr, reg, "conv")


def fmt(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


print(f"## Table S12: K50 by generation budget (K = 3-13, three seeds per target and regime, {len({r['target_id'] for r in P7})} targets)\n")
print("| Generations | Regime A, censored as failure | Regime A, complete case | Regime B, censored as failure | Regime B, complete case |")
print("|---|---|---|---|---|")
sets = [("200 (main experiment)", base), ("800", P7)] + ([("3200", P7b)] if len(P7b) == len(P7) else [])
for label, rows in sets:
    assert len(rows) == len(P7), (label, len(rows), len(P7))
    print(f"| {label} | {fmt(k50(rows, 'A', False))} | {fmt(k50(rows, 'A', True))} | {fmt(k50(rows, 'B', False))} | {fmt(k50(rows, 'B', True))} |")

print("\nPer-level convergence, Regime A / Regime B:\n")
print("| K | " + " | ".join(l for l, _ in sets) + " |")
print("|---|" + "---|" * len(sets))
for k in sorted({int(r["K_construction"]) for r in P7}):
    cells = []
    for _, rows in sets:
        c = []
        for reg in "AB":
            rr = [r for r in rows if int(r["K_construction"]) == k and r["regime"] == reg]
            c.append(f"{np.mean([r['converged'] == 'true' for r in rr]):.2f}")
        cells.append(" / ".join(c))
    print(f"| {k} | " + " | ".join(cells) + " |")

print("\n## Table S13: termination of non-converging runs, K <= 13 band, both regimes\n")
print("| Generations | n | Stagnated (early stop) | Budget exhausted | CPU-censored |")
print("|---|---|---|---|---|")
for label, rows in sets:
    f = [r for r in rows if r["converged"] != "true"]
    st = Counter(r["status"] for r in f)
    es = st.get("early_stopped", 0); be = st.get("budget_exhausted", 0); ce = sum(st.get(s, 0) for s in CENSORED)
    other = len(f) - es - be - ce
    print(f"| {label} | {len(f)} | {es/len(f):.0%} | {be/len(f):.0%} | {ce/len(f):.0%} |" + (f" (other statuses: {other})" if other else ""))

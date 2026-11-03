"""EML-family feasibility pilot report (D021) -> docs/results/eml_family_pilot_report.md

    .venv/bin/python -m reach.analysis.family_report

Pilot data: not confirmatory. All family grammars share one syntax (a binary operator, leaves {x, c}),
so syntactic spaces are identical and only the semantics differ. Minimal sizes are proven on EML-GP's
grid on (2, 3) under faithful semantics (no information-losing overflow or underflow).
"""
import json
import math
import time
from collections import defaultdict

import numpy as np

from reach.analysis.k50 import k50
from reach.analysis.pilot_report import Sizes, fmt_ci
from reach.family import CALC, CONSTANTS, FAMILY
from reach.paths import ROOT
from reach.runner.queue import Phase, fmt_dur

E0 = 100 * 200
CONST = {"F-EML": "1", "F-EDL": "e", "F-NEML": "−∞", "F-LDE0": "0", "F-LDE1": "1", "F-PLI": "1", "F-PLM": "1"}
FORM = {"F-EML": "eˣ − ln y", "F-EDL": "eˣ / ln y", "F-NEML": "ln x − eʸ", "F-LDE0": "ln x / eʸ",
        "F-LDE1": "ln x / eʸ", "F-PLI": "(ln x)^(1/y)", "F-PLM": "(ln x)^(−y)"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="scout_family")
    ap.add_argument("--out", default="docs/results/eml_family_pilot_report.md")
    a = ap.parse_args()
    present = [g for g in FAMILY if (ROOT / f"data/enumeration/{g}/counts.json").exists()]
    sizes = {g: Sizes(g) for g in present}
    L = ["# EML-family feasibility pilot report", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `reach.analysis.family_report` (phase `{a.phase}`). **Pilot data: not confirmatory.**",
         "", "Each grammar is one binary operator with leaves {x, c} (c = the operator's distinguished constant), so the "
         "syntactic spaces are identical and only the semantics differ. Faithful semantics: exact extended-real values "
         "are allowed (ln 0 = −∞, exp(−∞) = 0), information-losing overflow or underflow gives NaN. Minimal sizes are "
         "proven by enumeration on EML-GP's 100-point grid on (2, 3) (9-digit key) and re-verified by evaluation.", "",
         "## 1. Semantic growth", "",
         "| Operator | Form | c | Ceiling K | U(≤7) | U(≤11) | U(≤15) | U(≤ceiling) | Semantic bits/node | Viable share at ceiling |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for g in present:
        s = sizes[g]
        lv = json.loads((ROOT / f"data/enumeration/{g}/counts.json").read_text())["levels"]
        top = [l for l in lv if l["U"] > 0][-1]
        u = lambda k: f"{s.Ucum[k]:,}" if k in s.Ucum else "–"
        L.append(f"| {g} | {FORM[g]} | {CONST[g]} | {s.ceiling} | {u(7)} | {u(11)} | {u(15)} | {s.Ucum[s.ceiling]:,} "
                 f"| {s.bits_per_node:.2f} | {top.get('U_finite_nonconst', 0) / max(top['U'], 1):.2f} |")
    L += ["", "Viable share = fraction of the top level's new functions that are finite on the grid and non-constant.", ""]

    # 2. minimal constructions
    mins = {g: json.loads((ROOT / f"data/family/minimal_{g}.json").read_text())
            for g in present if (ROOT / f"data/family/minimal_{g}.json").exists()}
    if mins:
        gs = list(mins)
        L += ["## 2. Proven minimal size of calculator functions and constants", "",
              "| Function | " + " | ".join(gs) + " |", "|---|" + "---|" * len(gs)]
        for f in list(CALC) + list(CONSTANTS):
            cells = []
            for g in gs:
                h = mins[g]["table"].get(f)
                cells.append("–" if h is None else f"{h['K']}" + ("" if h["verified"] else "?"))
            L.append(f"| {f} | " + " | ".join(cells) + " |")
        found = {g: sum(h is not None for h in mins[g]["table"].values()) for g in gs}
        L += ["", "– = not found up to the operator's ceiling (" + ", ".join(f"{g} {mins[g]['ceiling']}" for g in gs)
              + "); ? = key matched but re-verification failed. Found: " + ", ".join(f"{g} {n}" for g, n in found.items()) + ".", ""]

    # 3. GP scout
    ph = Phase(a.phase)
    rows = list(ph.results().values()) if ph.manifest_path.exists() else []
    if rows:
        conv = lambda r: float(r.get("converged") == "true")
        exact = lambda r: float(r.get("exact_hash_match") == "true")
        by = defaultdict(list)
        for r in rows:
            by[(r["exp"], r["grammar"])].append(r)
        L += ["## 3. GP reach on each operator's own functions (scout, E₀ = 2×10⁴, 3 seeds × 10 targets per level)", "",
              "| Operator | K₅₀ nodes (conv) | K₅₀ nodes (exact) | K₅₀ semantic bits | Null K₅₀ nodes | GP − null (semantic bits) |",
              "|---|---|---|---|---|---|"]
        levels_tab = []
        for g in present:
            rr = by.get(("scout", g), [])
            if not rr:
                continue
            s = sizes[g]
            Ks = np.array([r["K"] for r in rr], float)
            cl = [r["target_id"] for r in rr]
            kc = k50(Ks, np.array([conv(r) for r in rr]), cl)
            ke = k50(Ks, np.array([exact(r) for r in rr]), cl)
            kb = k50([s.bits_sem(int(k)) for k in Ks], np.array([exact(r) for r in rr]), cl)
            gap = kb[0] - math.log2(E0) if not math.isnan(kb[0]) else float("nan")
            L.append(f"| {g} | {fmt_ci(kc)} | {fmt_ci(ke)} | {fmt_ci(kb)} | {s.null_k50_nodes(E0):.1f} | {gap:+.1f} |")
            cells = []
            for K in sorted({r["K"] for r in rr}):
                kr = [r for r in rr if r["K"] == K]
                cells.append(f"{K}{'' if K <= s.ceiling else '*'}: {np.mean([exact(r) for r in kr]):.2f}")
            levels_tab.append(f"| {g} | " + ", ".join(cells) + " |")
        L += ["", "K₅₀ on exact recovery is the primary outcome (D020). The null is size-ordered enumeration over distinct "
              "functions (D009).", "", "Exact-recovery rate by size (* = constructed size above the ceiling):", "",
              "| Operator | K: rate |", "|---|---|"] + levels_tab + [""]

        # 4. calculator targets
        calc = defaultdict(dict)
        for g in present:
            for r in by.get(("calc", g), []):
                calc[r["target_id"].split("_calc_", 1)[1]].setdefault(g, []).append(r)
        if calc:
            gs = [g for g in present if any(g in v for v in calc.values())]
            L += ["## 4. GP on the calculator functions (each at its proven minimal tree, 20 seeds, E₀)", "",
                  "Cell = exact-recovery rate (minimal K).", "", "| Function | " + " | ".join(gs) + " |", "|---|" + "---|" * len(gs)]
            common = [f for f in calc if all(g in calc[f] for g in gs)]
            for f in sorted(calc, key=lambda f: list(CALC).index(f) if f in CALC else 99):
                cells = []
                for g in gs:
                    rr = calc[f].get(g)
                    cells.append("–" if not rr else f"{np.mean([exact(r) for r in rr]):.2f} ({rr[0]['K']})")
                L.append(f"| {f} | " + " | ".join(cells) + " |")
            if common:
                L += ["", "Mean over the functions every operator expresses within its ceiling (" + ", ".join(common) + "): "
                      + ", ".join(f"{g} {np.mean([np.mean([exact(r) for r in calc[f][g]]) for f in common]):.2f}" for g in gs) + "."]
            L.append("")

        states = defaultdict(lambda: defaultdict(int))
        cpu = defaultdict(float)
        for r in rows:
            states[r["grammar"]][r["terminal_state"]] += 1
            cpu[r["grammar"]] += r["cpu_s"]
        L += ["## 5. Terminations and cost", "", "| Operator | Runs | Terminal states | CPU |", "|---|---|---|---|"]
        for g in present:
            if states[g]:
                L.append(f"| {g} | {sum(states[g].values())} | {', '.join(f'{k} {v}' for k, v in sorted(states[g].items()))} "
                         f"| {fmt_dur(cpu[g])} |")
    out = ROOT / a.out
    out.write_text("\n".join(L) + "\n")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()

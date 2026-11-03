"""Gate G2: calibration size levels from the pilot -> data/registration/levels.json (D062; docs/design/targets.md).

    .venv/bin/python -m reach.analysis.s2.g2_levels

Per calibration set, from the pilot (regime A, pilot targets and seeds; not confirmatory):
1. Budget levels (run to 16xE0): five sizes evenly spaced over the K50 band [K50(1/4 E0) - d, K50(16 E0) + d], with
   d = ln 3 / |b1| (half the E0 curve's 25-75% width), so that they bracket K50 at every budget. K50(16) is
   extrapolated from the pilot's K50 at 4xE0 and its nodes-per-doubling slope.
2. Main levels (E0 only): five sizes aimed evenly between the sizes where the E0 logistic curve (nodes) gives
   P = 0.95 and P = 0.05, among the sizes not taken in step 1. Where that band holds too few available sizes (narrow
   real-domain curves with odd sizes only), the nearest free sizes lie outside it, mostly in the tail where P(E0) is
   below 0.05; this happened for most real sets (D065).
Each size is snapped to the nearest unused available size: one the set can supply (at least MIN_U distinct functions
within the enumeration ceiling, or a constructed size above it, with the parity of the enumerated sizes, up to
2 * ceiling + 1: one operator on two minimal trees, the most `s2_targets.generate` can build; author, 2026-10-20).
Robustness sets (H8): per domain the anchor plus the full sets with the lowest and highest bits per node (lambda_G).
"""
import json
import math
from pathlib import Path

import numpy as np

MIN_U = 60                     # 30 targets per level with room for the real filters
N_LEVELS, N_BUDGET = 10, 5
ANCHORS = {"real": "F-EML", "bool": "BOOL-NAND-n5"}


def available_sizes(levels, kmax, min_u=MIN_U):
    nonempty = [st["K"] for st in levels if st["U"] > 0 and st["K"] >= 3]
    ceiling = max(nonempty)
    parities = {k % 2 for k in nonempty}
    inside = [st["K"] for st in levels if st["K"] >= 3 and st["U"] >= min_u]
    above = [k for k in range(ceiling + 1, min(kmax, 2 * ceiling + 1) + 1) if k % 2 in parities]   # one composition
    return sorted(set(inside) | set(above))


def _snap(av, targets, taken):
    out = []
    for w in targets:
        for k in av[np.argsort(np.abs(av - w), kind="stable")]:
            if int(k) not in taken and int(k) not in out:
                out.append(int(k))
                break
    return sorted(out)


def choose(available, b0, b1, k50_quarter, k50_16, n_main=N_LEVELS - N_BUDGET, n_budget=N_BUDGET):
    if b1 >= 0:
        raise ValueError("recovery must fall with size to place levels")
    k95, k05 = (math.log(19) - b0) / b1, (math.log(1 / 19) - b0) / b1
    out = choose_k(available, k95, k05, math.log(3) / abs(b1), k50_quarter, k50_16, n_main, n_budget)
    out["rule"].update(b0=b0, b1=b1)
    return out


def choose_k(available, k95, k05, d, k50_quarter, k50_16, n_main=N_LEVELS - N_BUDGET, n_budget=N_BUDGET):
    """The G2 rule from the curve's landmarks in nodes (used directly for held-out sets, whose curve comes from the
    frozen model in another unit)."""
    av = np.array(sorted(available))
    budget = _snap(av, np.linspace(k50_quarter - d, k50_16 + d, n_budget), [])
    main = _snap(av, np.linspace(k95, k05, n_main), budget)
    return dict(main=main, budget=budget, rule=dict(k95=k95, k05=k05, d=d, k50_quarter=k50_quarter, k50_16=k50_16))


def robust_sets(lam, anchor):
    others = {g: v for g, v in lam.items() if g != anchor}
    if len(others) < 2:
        raise ValueError("need at least two non-anchor sets")
    lo, hi = min(others, key=others.get), max(others, key=others.get)
    return [anchor, lo, hi]


def build(root=None):
    from reach import grammar as G
    from reach.analysis.k50 import logit
    from reach.analysis.s2.load import read_phase, tidy
    from reach.analysis.s2.phase1 import enum_dir
    from reach.paths import ROOT
    from reach.runner.phases import CAL_SETS, FULL_SETS
    root = Path(root or ROOT)
    pilot = json.loads((root / "data/analysis/pilot.json").read_text())
    lam = {s["set"]: s["bits_per_node"] for s in json.loads((root / "data/analysis/phase1.json").read_text())["sets"]}
    rows = read_phase("s2_pilot_reach_a", root) + read_phase("s2_pilot_reach_b", root)
    out = {}
    for g in CAL_SETS:
        e = pilot["sets"][g]
        e0 = tidy([r for r in rows if r.get("set") == g], (1,))
        b0, b1 = logit([r["K"] for r in e0], [r["y"] for r in e0])
        k4, rate = e["nodes"]["4"]["k50"], e["nodes_per_doubling"]
        k16 = k4 + 2 * max(rate, 0.0)
        counts = json.loads((enum_dir(g) / "counts.json").read_text())["levels"]
        kmax = int(math.ceil(max((math.log(1 / 19) - b0) / b1, k16 + math.log(3) / abs(b1)))) + 20   # room for the tail
        lv = choose(available_sizes(counts, kmax), float(b0), float(b1), e["nodes"]["0.25"]["k50"], k16)
        lv["rule"]["nodes_per_doubling"] = rate
        out[g] = lv
    out["_robust"] = {d: robust_sets({g: lam[g] for g in FULL_SETS if G.get(g).kind == d}, a)
                      for d, a in ANCHORS.items()}
    out["_source"] = "reach.analysis.s2.g2_levels from the pilot (pilot targets and seeds; not confirmatory), D062"
    return out


def main():
    from reach.paths import ROOT
    p = ROOT / "data/registration/levels.json"
    res = build()
    p.write_text(json.dumps(res, indent=1, default=float))
    for g, v in res.items():
        if not g.startswith("_"):
            print(f"{g:15s} budget {v['budget']}  main {v['main']}")
    print("robust", res["_robust"])


if __name__ == "__main__":
    main()

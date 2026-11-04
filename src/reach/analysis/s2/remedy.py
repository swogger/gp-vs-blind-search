"""The real-domain loss beside its remedies (D100; REQ-ANA-93, 94). EXPLORATORY and descriptive on registered records;
no test, no decision. Stage 4's registered analysis will use the same restart estimator.

    python -m reach.analysis.s2.remedy [--boot B]

1. Restart curve on Stage 1's allocation runs (real sets): K50 per budget and run length, exact or plug-in, bracketed
   or extrapolated, beside one long run and the exact null; rates per doubling of restarts on bracketed exact points.
2. Gaps to the exact null (K50 - log2 E) for tree GP, Cartesian GP and Operon, and per H8 configuration; Boolean sizes
   capped at the space bound (guardrail row 154).
3. Evaluation ratios: GP needs 2^(-gap) times blind search's evaluations for the same reach.
Writes data/analysis/remedy.json and docs/status/remedy_report.md.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.s2 import alloc, confirm, law
from reach.analysis.s2.load import E0, outcome, read_phase, tidy

BUDGETS = (1, 2, 4, 8, 16)
LENGTHS = (25, 50, 100, 200)
GENS_E0 = 200            # generations per E0 at population 100


def eval_ratio(gap):
    """The evaluations GP spent, as a multiple of what the null needs to reach GP's own reach: 2^(log2 E - K50), since the
    null reaches K50 with 2^K50 evaluations."""
    return 2.0 ** (-gap)


# ---------------------------------------------------------------- 1. restart curve
def restart_curve(runs_by_target, size_of, L, budgets=BUDGETS):
    """`runs_by_target` {target: [runs of at least L generations]}, `size_of` {target: bits_sem}. Per budget: R
    restarts of L generations, K50 of the per-target probabilities (`alloc._k50`), the plug-in and extrapolation
    flags, and the lower bound (the top tested level when extrapolated)."""
    top = max(size_of[t] for t in runs_by_target)
    out = {}
    for B in budgets:
        R = int(round(B * GENS_E0 / L))
        sizes, ps, plug = [], [], False
        for t, runs in runs_by_target.items():
            hits = [outcome(r, L) == "success" for r in runs if r.get("terminal_state") != "error"]
            p, pl = alloc.p_at_least_one(sum(hits), len(hits), R)
            plug |= pl
            sizes.append(size_of[t])
            ps.append(p)
        k = alloc._k50(sizes, ps)
        ext = bool(k > top) if not math.isnan(k) else False
        out[B] = dict(R=R, plug_in=plug, k50=k, extrapolated=ext, lower_bound=top if ext else k,
                      null=math.log2(B * E0))
    return out


def rate_inside(curve):
    """Least-squares K50 per doubling of the budget over bracketed, exact points (at least 3), else NaN."""
    pts = [(math.log2(B), v["k50"]) for B, v in sorted(curve.items())
           if not v["extrapolated"] and not v["plug_in"] and np.isfinite(v["k50"])]
    if len(pts) < 3:
        return dict(rate=float("nan"), points=len(pts))
    x, y = np.array(pts).T
    return dict(rate=float(np.polyfit(x, y, 1)[0]), points=len(pts))


# ---------------------------------------------------------------- 2. gaps to the exact null
def null_gaps(recs, budget):
    """Per set: K50 (bits_sem; Boolean sizes capped at the space bound) of the records at `budget`, minus log2 of the
    budget's evaluations; flagged when K50 lies above the top tested level (then the gap is a lower bound) or above the
    space bound (Boolean: K50 reported at the bound)."""
    from reach import grammar as G
    from reach.units import capped, space_bits
    by = defaultdict(list)
    for r in recs:
        if r["budget"] == budget:
            g = G.get(r["set"])
            by[r["set"]].append(dict(r, units=dict(r["units"], bits_sem=capped(r["units"]["bits_sem"], g))))
    out = {}
    for s, rr in sorted(by.items()):
        k = law._k50(rr, "bits_sem")
        bound = space_bits(G.get(s))
        top = max(r["units"]["bits_sem"] for r in rr)
        above_space = bool(np.isfinite(k) and k > bound)
        k = bound if above_space else k
        null = math.log2(budget * E0)
        out[s] = dict(k50=k, null=null, gap=k - null, extrapolated=bool(np.isfinite(k) and k > top),
                      capped_above=above_space, n=len(rr), share=float(np.mean([r["y"] for r in rr])))
    return out


# ---------------------------------------------------------------- 4. H11 (Stage 4; REQ-ANA-95; docs/design/remedy.md)
H11_R, H11_L, H11_BUDGET = 32, 100, 16         # 32 restarts of 100 generations = 16 x E0
H11_MARGIN, H11_GAIN = -1.0, 2.0
H11_BOOT, H11_SEED = 2000, 20261020


def hits(runs, L):
    """(s, n): runs exact within L generations, and runs counted. An error row is left out of n; a row killed without a
    result, or stopped on its CPU limit before L, counts as a failure (`load.outcome`)."""
    ok = [r for r in runs if r.get("terminal_state") != "error"]
    return sum(outcome(r, L) == "success" for r in ok), len(ok)


def restart_k50(runs_by_target, size_of, R=H11_R, L=H11_L):
    """K50 of R restarts of L generations, P per target from all its runs (`alloc.p_at_least_one`). Bracketing rule: a
    K50 above the top tested level (including +inf) is replaced by the top level and flagged; NaN stays NaN."""
    top = max(size_of[t] for t in runs_by_target)
    sizes, ps = [], []
    for t, runs in runs_by_target.items():
        s, n = hits(runs, L)
        ps.append(alloc.p_at_least_one(s, n, R)[0])
        sizes.append(size_of[t])
    raw = alloc._k50(sizes, ps)
    ext = bool(not math.isnan(raw) and raw > top)
    return dict(k50=top if ext else raw, raw=raw, extrapolated=ext, top=top)


def conjunction_p(ps):
    """p of 'every member holds': the largest member p."""
    return max(ps)


def p_beyond(reps, margin, side, boot):
    """p for 'the statistic lies beyond `margin`' (side 'above': > margin): (replicates at or on the wrong side of the
    margin + 1)/(kept + 1); estimable when at least 90% of `boot` replicates are kept."""
    ok = [x for x in reps if np.isfinite(x)]
    bad = sum(x <= margin for x in ok) if side == "above" else sum(x >= margin for x in ok)
    lo, hi = (float(v) for v in np.percentile(ok, [5, 95])) if ok else (float("nan"),) * 2
    return dict(p=(bad + 1) / (len(ok) + 1), kept=len(ok), estimable=len(ok) >= 0.9 * boot, lo90=lo, hi90=hi)


def decide(tests, alpha=0.05):
    """Holm over the family, as Stage 3 (`h6d.holm`): tests sorted by p, a test that is not estimable counting as
    p = 1; each compared with alpha / (m - i) (p <= threshold); the first failure stops the procedure (guardrail row
    155: the first version stopped at a non-estimable test whatever its p)."""
    eff = {k: tests[k]["p"] if tests[k]["estimable"] else 1.0 for k in tests}
    order = sorted(tests, key=lambda k: eff[k])
    out, going = {}, True
    for i, k in enumerate(order):
        thr = alpha / (len(order) - i)
        going = going and eff[k] <= thr
        out[k] = dict(tests[k], threshold=thr, supported=bool(going))
    return out


def h11(restart, one, boot=H11_BOOT, seed=H11_SEED):
    """`restart` {set: ({target: runs}, {target: bits})}; `one` {set: [one-run records at 16 x E0 with target_id, y,
    units]}. Returns the point estimates, both tests and the per-set values (REQ-ANA-95)."""
    E = H11_BUDGET * E0
    null = math.log2(E)
    sets = sorted(restart)

    def stats(pick):
        per = {}
        for g in sets:
            runs, size = restart[g]
            ids = pick(g, sorted(set(runs) | {r["target_id"] for r in one[g]}))
            rb = {f"{t}#{i}": runs[t] for i, t in enumerate(ids) if t in runs}
            sz = {f"{t}#{i}": size[t] for i, t in enumerate(ids) if t in runs}
            k = restart_k50(rb, sz)
            by = defaultdict(list)
            for r in one[g]:
                by[r["target_id"]].append(r)
            k1 = law._k50([r for t in ids for r in by.get(t, [])], "bits_sem")
            per[g] = dict(k50=k["k50"], raw=k["raw"], extrapolated=k["extrapolated"], one_run=k1,
                          gap=k["k50"] - null, gain=k["k50"] - k1)
        return per
    point = stats(lambda g, ids: ids)
    rng = np.random.default_rng(seed)
    d_reps, gain_reps = [], {g: [] for g in sets}
    for _ in range(boot):
        rep = stats(lambda g, ids: [ids[j] for j in rng.integers(0, len(ids), len(ids))])
        gaps = [v["gap"] for v in rep.values()]
        d_reps.append(float(np.mean(gaps)) if all(np.isfinite(gaps)) else float("nan"))
        for g in sets:
            gain_reps[g].append(rep[g]["gain"])
    D = float(np.mean([v["gap"] for v in point.values()]))
    t1 = dict(p_beyond(d_reps, H11_MARGIN, "above", boot), estimate=D, margin=H11_MARGIN)
    t1["estimable"] = t1["estimable"] and math.isfinite(D)
    per_gain = {g: p_beyond(gain_reps[g], H11_GAIN, "above", boot) for g in sets}
    t2 = dict(p=conjunction_p([v["p"] for v in per_gain.values()]), margin=H11_GAIN,
              estimable=all(v["estimable"] for v in per_gain.values())
              and all(math.isfinite(point[g]["gain"]) for g in sets),
              estimate=min(point[g]["gain"] for g in sets))
    gaps = [v["gap"] for v in point.values()]
    se = float(np.std(gaps, ddof=1) / math.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")   # undefined for 1 set
    return dict(null=null, per_set=point, per_set_gain_tests=per_gain,
                tests=decide({"H11-1": t1, "H11-2": t2}),
                set_level=dict(mean=D, se=se, n_sets=len(gaps)), boot=boot)


# ---------------------------------------------------------------- driver
def _tidy(root, phases, targets, budgets, keep=lambda r: True):
    """Tidy records with units; the two 6-input size-check sets are left out, as in H6c's main sets."""
    rows = [r for ph in phases for r in read_phase(ph, root) if keep(r) and r.get("set") not in confirm.SIZE_CHECKS]
    sets = sorted({r["set"] for r in rows})
    confirm._load_units(root, sets, targets)
    return [r for r in confirm._with_units(tidy(rows, budgets)) if r.get("units", {}).get("bits_sem") is not None]


def _domain_means(gaps, real):
    """Mean gap over the domain's sets, as text: "≥" when any set's K50 is a lower bound (above its top level or at
    the space cap), "n/a (saturated)" when no set is estimable."""
    dom = [x for s, x in gaps.items() if (not s.startswith("BOOL")) == real]
    v = [x for x in dom if np.isfinite(x["gap"])]
    if not v:
        return "n/a (saturated)"
    bound = any(x["extrapolated"] or x["capped_above"] for x in v) or len(v) < len(dom)
    return ("≥ " if bound else "") + _f(float(np.mean([x["gap"] for x in v])), sign=True)


def build(root):
    root = Path(root)
    res = dict(label="EXPLORATORY / descriptive on registered records (D100); not confirmatory", restarts={},
               systems={}, configs={})
    # 1. restarts on the allocation runs (real sets)
    a_rows = read_phase("s2_cal_a0", root) + read_phase("s2_cal_a16", root)
    al_rows = read_phase("s2_cal_alloc", root)
    real = sorted({r["set"] for r in al_rows if not r["set"].startswith("BOOL")})
    confirm._load_units(root, real, "calibration")
    size_of = {(r["set"], r["target_id"]): r["units"]["bits_sem"] for r in confirm._with_units(
        [dict(set=r["set"], target_id=r["target_id"], K=r.get("K_construction", r["K"]), units=r.get("units") or {},
              y=0, budget=1) for r in al_rows if r["set"] in real])}
    for g, (_tids, runs) in sorted(confirm.alloc_runs(a_rows, al_rows).items()):
        if g not in real:
            continue
        by = defaultdict(list)
        for r in runs:
            by[r["target_id"]].append(r)
        sz = {t: size_of[(g, t)] for t in by}
        curves = {L: restart_curve(by, sz, L) for L in LENGTHS}
        one = [dict(set=g, target_id=r["target_id"], budget=B, y=int(outcome(r, B * GENS_E0) == "success"),
                    units={"bits_sem": sz[r["target_id"]]})
               for r in a_rows if r["set"] == g and r["target_id"] in by and r.get("generations") == 16 * GENS_E0
               and r.get("terminal_state") != "error" for B in BUDGETS]
        top = max(sz.values())
        one_k = {B: law._k50([x for x in one if x["budget"] == B], "bits_sem") for B in BUDGETS}
        res["restarts"][g] = dict(levels=sorted({round(v, 2) for v in sz.values()}), top=top,
                                  curves={L: {str(B): v for B, v in c.items()} for L, c in curves.items()},
                                  rates={L: rate_inside(c) for L, c in curves.items()},
                                  one_run={str(B): dict(k50=k, extrapolated=bool(np.isfinite(k) and k > top))
                                           for B, k in one_k.items()})
    # 2. three systems against the exact null
    tree_cal = _tidy(root, ("s2_cal_a0", "s2_cal_a16"), "calibration", (1, 16),
                     lambda r: r.get("algorithm") == "A" and r.get("config") == "base")
    tree_ho = _tidy(root, ("s2_ho_a0", "s2_ho_a16"), "heldout", (1, 16))
    for B in (1, 16):
        res["systems"][f"tree GP, {B} x E0"] = null_gaps(tree_cal + tree_ho, B)
    reg = json.loads((root / "data/analysis/calibration.json").read_text())["h6c"]["gaps"]["real"]["per_set"]
    got = res["systems"]["tree GP, 1 x E0"]
    bad = {g: (got[g]["gap"], v) for g, v in reg.items() if not math.isclose(got[g]["gap"], v, abs_tol=1e-9)}
    if bad:
        raise RuntimeError(f"instrument check: real E0 gaps differ from the registered H6c per-set gaps: {bad}")
    for cfg in ("n100", "n2000"):
        cg = _tidy(root, ("s2_cal_cgp",), "calibration", (1, 16), lambda r: r.get("config") == cfg) + \
            _tidy(root, ("s2_ho_cgp",), "heldout", (1, 16), lambda r: r.get("config") == cfg)
        for B in (1, 16):
            res["systems"][f"Cartesian GP {cfg}, {B} x E0"] = null_gaps(cg, B)
    for cfg in ("symbolic", "default"):
        op = _tidy(root, ("s2_cal_operon",), "calibration", (16,), lambda r: r.get("config") == cfg) + \
            _tidy(root, ("s2_ho_operon",), "heldout", (16,), lambda r: r.get("config") == cfg)
        res["systems"][f"Operon {cfg}, 16 x E0"] = null_gaps(op, 16)
    # 3. H8 configurations (calibration; budgets 1 and 4 x E0)
    rob = _tidy(root, ("s2_cal_robust",), "calibration", (1, 4))
    base = [r for r in _tidy(root, ("s2_cal_a0", "s2_cal_a16"), "calibration", (1, 4),
                             lambda r: r.get("algorithm") == "A" and r.get("config") == "base")
            if r["set"] in {x["set"] for x in rob}]
    for cfg in ["base"] + sorted({r["config"] for r in rob}):
        rr = base if cfg == "base" else [r for r in rob if r["config"] == cfg]
        res["configs"][cfg] = {str(B): null_gaps(rr, B) for B in (1, 4)}
    res["built"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return res


def _f(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


CONFIG_NAMES = dict(base="base (pop 100, tournament 7)", R1="R1 population 25", R2="R2 population 1,000",
                    R3="R3 tournament 2", R4="R4 lexicase", R5="R5 crossover 0.5", R6="R6 depth limit 17")


def render(res):
    L = ["# The real-domain loss beside its remedies (EXPLORATORY)", "",
         f"> Generated by `reach.analysis.s2.remedy` ({res['built']}); every number from data/analysis/remedy.json. Do "
         "not edit by hand. Descriptive on registered records (D100): no test, no decision; not confirmatory.", "",
         "## 1. Restarts (Stage 1 allocation runs, real sets, 100 targets x 32 seeds)", "",
         "K₅₀ (bits_sem) per budget and run length L (R restarts of L generations). Flags: `p` plug-in (R > 32 "
         "seeds: biased down); `^` above the top tested level (extrapolated; the top level is a lower bound). The "
         "null's reach is log₂ E exactly.", ""]
    for g, v in res["restarts"].items():
        L += [f"**{g}** (levels {', '.join(f'{x:.1f}' for x in v['levels'])} bits)", "",
              "| budget | null | one run | " + " | ".join(f"L {x}" for x in LENGTHS) + " |",
              "|---|---|---|" + "---|" * len(LENGTHS)]
        for B in map(str, BUDGETS):
            o = v["one_run"][B]
            cells = []
            for Lx in LENGTHS:
                c = v["curves"][Lx][B]
                cells.append(f"{_f(c['k50'])}{'^' if c['extrapolated'] else ''}{'p' if c['plug_in'] else ''} "
                             f"(R {c['R']})")
            L.append(f"| {B} × E₀ | {_f(v['curves'][LENGTHS[0]][B]['null'])} | {_f(o['k50'])}"
                     f"{'^' if o['extrapolated'] else ''} | " + " | ".join(cells) + " |")
        L += ["", "Bits per doubling of restarts (bracketed exact points): " + "; ".join(
            f"L {Lx}: {_f(v['rates'][Lx]['rate'])} ({v['rates'][Lx]['points']} points)" for Lx in LENGTHS), ""]
    L += ["## 2. Three GP systems against the exact null", "",
          "Gap = K₅₀ − log₂ E (bits); per domain the mean over sets, the range and the number of sets below 0 (real) or "
          "above 0 (Boolean). The main and held-out sets (the 6-input size checks left out, as in H6c). Boolean sizes "
          "capped at 32 bits; `^` sets whose K₅₀ lies above their top tested level (gap a lower bound).", "",
          "| system and budget | real: mean [min, max], sets below 0 | Boolean: mean [min, max], sets above 0 |",
          "|---|---|---|"]
    for name, gaps in res["systems"].items():
        cells = []
        for real in (True, False):
            dom = {s: x for s, x in gaps.items() if (not s.startswith("BOOL")) == real}
            v = {s: x for s, x in dom.items() if np.isfinite(x["gap"])}
            if not dom:
                cells.append("–")
                continue
            gs = [x["gap"] for x in v.values()]
            n = sum(x < 0 for x in gs) if real else sum(x > 0 for x in gs)
            flags = [f"{sum(x['extrapolated'] for x in v.values())}^", f"{sum(x['capped_above'] for x in v.values())} "
                     "at the 32-bit cap", f"{len(dom) - len(v)} not estimable (saturated)"]
            flags = [f for f in flags if not f.startswith("0")]
            cells.append((f"{_f(float(np.mean(gs)), sign=True)} [{_f(min(gs), sign=True)}, {_f(max(gs), sign=True)}], "
                          f"{n} of {len(gs)}" if gs else "n/a") + (f" ({'; '.join(flags)})" if flags else ""))
        L.append(f"| {name} | {cells[0]} | {cells[1]} |")
    L += ["", "## 3. H8 configurations against the exact null (calibration; 3 sets per domain)", "",
          "| configuration | real gap at E₀ / 4 × E₀ (mean of 3 sets) | Boolean gap at E₀ / 4 × E₀ (capped) |",
          "|---|---|---|"]
    for cfg, by in res["configs"].items():
        L.append(f"| {CONFIG_NAMES.get(cfg, cfg)} | {_domain_means(by['1'], True)} / {_domain_means(by['4'], True)} | "
                 f"{_domain_means(by['1'], False)} / {_domain_means(by['4'], False)} |")
    L += ["", "## 4. Evaluation ratios (Kronberger et al.'s currency)", "",
          "Evaluations GP needs for exact blind search's reach, as a multiple of blind search's: 2^(−gap).", "",
          "| real domain | one run, E₀ | one run, 16 × E₀ | best bracketed restarts (L 100), 16 × E₀ |", "|---|---|---|---|"]
    one = res["systems"]["tree GP, 1 x E0"]
    one16 = res["systems"]["tree GP, 16 x E0"]
    for g, v in res["restarts"].items():
        c = v["curves"][100]["16"]
        rs = (f"≤ ×{eval_ratio(c['lower_bound'] - c['null']):.1f} (lower bound on reach)" if c["extrapolated"]
              else f"×{eval_ratio(c['k50'] - c['null']):.1f}")
        L.append(f"| {g} | ×{eval_ratio(one[g]['gap']):.1f} | ×{eval_ratio(one16[g]['gap']):.1f} | {rs} |")
    return "\n".join(L) + "\n"


def main(argv=None, root=None):
    from reach.paths import ROOT
    argparse.ArgumentParser().parse_args(argv)
    root = Path(root or ROOT)
    res = build(root)
    (root / "data/analysis/remedy.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (root / "docs/status/remedy_report.md").write_text(render(res))
    print("written data/analysis/remedy.json and docs/status/remedy_report.md")


if __name__ == "__main__":
    main()

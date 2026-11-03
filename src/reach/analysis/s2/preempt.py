"""Pre-emptive analyses for the paper (D099; REQ-ANA-89 to 91). EXPLORATORY: existing data, no new runs, no test, no
Holm family, no decision; never confirmatory.

    python -m reach.analysis.s2.preempt [--boot B]

1. Set offsets out of sample: each held-out set's intercept at E0 with the calibration slope fixed, predicted by the
   property regression chosen on the calibration sets (data/analysis/calibration_exploratory.json).
2. Weaker nulls: H6c's gap against the bottom-up enumerator (duplicates included) and uniform random search, beside
   the primary null; and each null's reach per budget doubling.
3. Tolerance reach (real domain): K50 with success = validation nMSE <= 1e-6 (`gen_tol`) beside exact recovery. In the
   Boolean domain the tolerance is 0, so the two are the same outcome (checked on the records).
Writes data/analysis/preempt.json and docs/status/preempt_report.md.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from reach.analysis.s2 import confirm, law, null, nullgap
from reach.analysis.s2.load import E0, read_phase, tidy

SEED = 20261102
BUDGETS = (1, 2, 4, 8, 16)
KINDS = ("enumeration", "bottom_up", "uniform")


# ---------------------------------------------------------------- 3. tolerance outcome
def tidy_tol(rows, budgets, censored_as_failure=True):
    """`load.tidy` with success read from `gen_tol` (first generation within tolerance) instead of `gen_exact`. The
    engine checks exactness only once the best tree is within tolerance, so gen_tol <= gen_exact; a row that breaks
    this is refused."""
    for r in rows:
        ge, gt = r.get("gen_exact"), r.get("gen_tol")
        if ge is not None and (gt is None or gt > ge):
            raise ValueError(f"{r.get('key', r.get('target_id'))}: tolerance generation {gt} after exact {ge}")
    return tidy([{**r, "gen_exact": r.get("gen_tol")} for r in rows], budgets, censored_as_failure)


# ---------------------------------------------------------------- 2. weaker nulls
def null_shift(levels_by_set, E, kind):
    """Domain mean over sets of log2 E minus the null's reach in bits_sem: added to H6c's gap (GP's K50 minus log2 E) it
    gives the gap against that null."""
    return float(np.mean([math.log2(E) - null.reach_bits_sem(lv, E, kind) for lv in levels_by_set.values()]))


def gap_against(g6, levels_by_set, E, kind):
    """H6c's registered gap and CI (`g6`: gap, lo, hi) against another null: shifted by the domain mean of (primary
    null's reach - that null's reach), both computed, so the computed reach's level granularity cancels and the
    primary null gives the registered gap exactly."""
    sh = 0.0 if kind == "enumeration" else null_shift(levels_by_set, E, kind) - null_shift(levels_by_set, E,
                                                                                          "enumeration")
    return dict(gap=g6["gap"] + sh, lo=g6["lo"] + sh, hi=g6["hi"] + sh, shift=sh)


def null_rate(levels_by_set, kind, budgets=BUDGETS):
    """Domain mean over sets of the null's reach (bits_sem) per budget doubling, least squares over `budgets` x E0."""
    x = [math.log2(b) for b in budgets]
    return float(np.mean([np.polyfit(x, [null.reach_bits_sem(lv, b * E0, kind) for b in budgets], 1)[0]
                          for lv in levels_by_set.values()]))


# ---------------------------------------------------------------- 1. set offsets out of sample
def _alpha(s, y, beta):
    """Maximum-likelihood intercept of logit P = alpha - beta s with beta fixed; NaN when every run has one outcome (no
    finite value). Bisection on the score sum(y - p), which falls monotonically in alpha: Newton from 0 overshot on
    held-out data and divided by a zero curvature (guardrail row 153)."""
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    score = lambda a: float((y - 1 / (1 + np.exp(-(a - beta * s)))).sum())   # noqa: E731
    lo, hi = -1.0, 1.0
    while score(lo) < 0:
        lo *= 2
    while score(hi) > 0:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if score(mid) > 0 else (lo, mid)
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2


def alpha_fixed_beta(recs, beta, unit, boot=200, seed=SEED):
    """A set's intercept at a fixed slope, with a 95% target-bootstrap CI (dropped replicates counted)."""
    s = np.array([law._size(r, unit) for r in recs], float)
    y = np.array([r["y"] for r in recs], float)
    by = law._by_target([dict(r, _i=i) for i, r in enumerate(recs)])
    ids, rng = list(by), np.random.default_rng(seed)
    reps = []
    for _ in range(boot):
        ix = [r["_i"] for j in rng.integers(0, len(ids), len(ids)) for r in by[ids[j]]]
        reps.append(_alpha(s[ix], y[ix], beta))
    ok = [x for x in reps if np.isfinite(x)]
    lo, hi = (float(v) for v in np.percentile(ok, [2.5, 97.5])) if ok else (float("nan"),) * 2
    return dict(est=_alpha(s, y, beta), lo=lo, hi=hi, dropped=boot - len(ok), targets=len(ids))


def offset_prediction(alpha_obs, feats, chosen, cal_mean):
    """Held-out intercepts predicted by the calibration regression (`chosen`: properties and coefficients, intercept
    first), scored by RMSE against predicting every set at the calibration mean."""
    c = chosen["coef"]
    pred = {g: float(c[0] + sum(k * feats[g][p] for k, p in zip(c[1:], chosen["properties"]))) for g in alpha_obs}
    gs = sorted(alpha_obs)
    rm = lambda e: float(math.sqrt(np.mean(np.square(e))))   # noqa: E731
    return dict(pred=pred, rmse_chosen=rm([alpha_obs[g] - pred[g] for g in gs]),
                rmse_mean=rm([alpha_obs[g] - cal_mean for g in gs]),
                corr=float(np.corrcoef([alpha_obs[g] for g in gs], [pred[g] for g in gs])[0, 1]) if len(gs) > 2 else None)


# ---------------------------------------------------------------- 4. Boolean size bounds
def bool_size_variants(recs, constructed, ceilings, parities, units_fn):
    """The four size readings of REQ-ANA-92 as new record lists (the input records are never changed): registered;
    capped at the space bound; constructed sizes at their lower bound (`confirm.lower_bound_sizes`), capped; proven
    targets only."""
    from reach import grammar as G
    from reach.units import capped

    def cap(rr):
        return [dict(r, units=dict(r["units"], bits_sem=capped(r["units"]["bits_sem"], G.get(r["set"])))) for r in rr]
    return dict(registered=[dict(r) for r in recs], capped=cap(recs),
                lower_bound=cap(confirm.lower_bound_sizes(recs, constructed, ceilings, parities, units_fn)),
                proven_only=[dict(r) for r in recs if (r["set"], r["target_id"]) not in constructed])


# ---------------------------------------------------------------- driver
def _recs(root, phases, sets, targets, budgets, tol=False):
    rows = [r for ph in phases for r in read_phase(ph, root) if r.get("set") in sets]
    rows = [r for r in rows if r.get("algorithm") == "A" and r.get("config") == "base"]
    confirm._load_units(root, sets, targets)
    return rows, confirm._with_units((tidy_tol if tol else tidy)(rows, budgets))


def build(root, boot=1000):
    root = Path(root)
    model = json.loads((root / "data/registration/h7_model.json").read_text())
    preds = json.loads((root / "data/registration/heldout_predictions.json").read_text())
    cal = json.loads((root / "data/analysis/calibration.json").read_text())
    expl = json.loads((root / "data/analysis/calibration_exploratory.json").read_text())["intercepts"]
    res = dict(label="EXPLORATORY (D099; existing data, no new runs; not confirmatory)", boot=boot,
               offsets={}, nulls={}, tolerance={})
    for d in ("real", "bool"):
        main = model[d]["fit_records"]["sets"]
        # 1. offsets
        ex = expl[d]
        ho = sorted(g for g in preds["features"] if g.startswith("BOOL") == (d == "bool"))
        _, rr = _recs(root, ("s2_ho_a0", "s2_ho_a16"), ho, "heldout", (1,))
        a = {g: alpha_fixed_beta([r for r in rr if r["set"] == g], ex["beta"], ex["unit"], boot // 5)
             for g in ho}
        feats = confirm.set_features(root, ho)
        cal_alpha = {g: v["est"] for g, v in ex["alpha"].items() if math.isfinite(v["est"])}
        sc = offset_prediction({g: v["est"] for g, v in a.items()}, feats, ex["chosen"],
                               float(np.mean(list(cal_alpha.values()))))
        res["offsets"][d] = dict(unit=ex["unit"], beta=ex["beta"], chosen=ex["chosen"], calibration=cal_alpha,
                                 heldout=a, features={g: {p: feats[g][p] for p in ex["chosen"]["properties"]}
                                                      for g in ho}, **sc)
        # 2. weaker nulls (H6c's registered gap and CI shifted)
        lv = confirm._null_levels(main)
        g6 = cal["h6c"]["gaps"][d]
        res["nulls"][d] = {k: dict(gap_against(g6, lv, E0, k), rate=(rt := null_rate(lv, k)), per_bit=2 ** (1 / rt))
                           for k in KINDS}
        res["nulls"][d]["registered"] = dict(gap=g6["gap"], lo=g6["lo"], hi=g6["hi"])
        # 3. tolerance
        rows, ex_recs = _recs(root, ("s2_cal_a0", "s2_cal_a16"), main, "calibration", BUDGETS)
        if d == "bool":
            same = sum(r.get("gen_tol") == r.get("gen_exact") for r in rows)
            res["tolerance"][d] = dict(rows=len(rows), same=same, note="tolerance 0: the same outcome as exact")
            continue
        _, tol_recs = _recs(root, ("s2_cal_a0", "s2_cal_a16"), main, "calibration", BUDGETS, tol=True)
        per = {}
        for g in main:
            te = [r for r in tol_recs if r["set"] == g]
            ee = [r for r in ex_recs if r["set"] == g]
            t1, e1 = [r for r in te if r["budget"] == 1], [r for r in ee if r["budget"] == 1]
            rt = law.reach_per_doubling(te, "bits_sem", BUDGETS, boot // 5)
            per[g] = dict(k50_tol=nullgap._k50_bits(t1), k50_exact=nullgap._k50_bits(e1),
                          share_tol=float(np.mean([r["y"] for r in t1])),
                          share_exact=float(np.mean([r["y"] for r in e1])), rate_tol=rt,
                          rate_exact=cal["h6"][g]["bits_per_doubling"])
        dg = nullgap.domain_gap({g: [r for r in tol_recs if r["set"] == g and r["budget"] == 1] for g in main},
                                E0, boot)
        dg.pop("_reps", None)
        res["tolerance"][d] = dict(per_set=per, gap_tol=dg, gap_exact=res["nulls"][d]["registered"])
    res["bool_bounds"] = bool_bounds(root, model["bool"]["fit_records"]["sets"], cal, boot)
    res["built"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return res


def bool_bounds(root, sets, cal, boot):
    """REQ-ANA-92: the Boolean H6c gap at E0 (domain mean, target bootstrap), H6d's G per set (K50 of GP minus the
    registered twin's) and the budget rates under the four size readings. The registered reading must reproduce the
    registered gap (instrument check)."""
    from reach.analysis.s2.confirm import _sensitivity_inputs
    from reach import grammar as G
    from reach.units import space_bits
    h6d = json.loads((Path(root) / "data/analysis/h6d.json").read_text())["per_set"]["bool"]
    _, recs = _recs(root, ("s2_cal_a0", "s2_cal_a16"), sets, "calibration", BUDGETS)
    var = bool_size_variants(recs, *_sensitivity_inputs(root, sets))
    cap = space_bits(G.get(sets[0]))
    out = {}
    for name, rr in var.items():
        e0 = {g: [r for r in rr if r["set"] == g and r["budget"] == 1] for g in sets}
        dg = nullgap.domain_gap(e0, E0, boot)
        dg.pop("_reps", None)
        per = {}
        for g in sets:
            k = nullgap._k50_bits(e0[g])
            ks = [nullgap._k50_bits([r for r in rr if r["set"] == g and r["budget"] == b]) for b in BUDGETS]
            per[g] = dict(k50_e0=k, G=k - h6d[g]["k50_twin"],
                          rate=law.reach_per_doubling([r for r in rr if r["set"] == g], "bits_sem", BUDGETS,
                                                      boot // 5),
                          budgets_above_space=[b for b, x in zip(BUDGETS, ks) if np.isfinite(x) and x > cap])
        out[name] = dict(gap=dg, G_mean=float(np.mean([v["G"] for v in per.values()])), per_set=per)
    reg = cal["h6c"]["gaps"]["bool"]["gap"]
    if not math.isclose(out["registered"]["gap"]["gap"], reg, abs_tol=1e-9):
        raise RuntimeError(f"instrument check: registered reading gives {out['registered']['gap']['gap']}, "
                           f"registered gap {reg}")
    return dict(space_bits=cap, readings=out)


def _f(x, d=2, sign=False):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:+.{d}f}" if sign else
                                                                                       f"{x:.{d}f}")


def render(res):
    L = ["# Pre-emptive analyses (EXPLORATORY)", "",
         f"> Generated by `reach.analysis.s2.preempt` ({res['built']}); every number from data/analysis/preempt.json. "
         "Do not edit by hand. EXPLORATORY (D099): existing data, no new runs, no test, no decision; not confirmatory. "
         f"Bootstrap {res['boot']} replicates (offsets and tolerance rates {res['boot'] // 5}).", "",
         "## 1. Set offsets out of sample", "",
         "Held-out sets' intercepts at E₀ (logit P = α − β s, β fixed at the calibration value), predicted by the "
         "property regression chosen on the calibration sets (Stage 1 exploratory; leave-one-set-out rule, D085), "
         "against predicting the calibration mean.", ""]
    for d, o in res["offsets"].items():
        ch = o["chosen"]
        L += [f"**{d}** (unit {o['unit']}, β {o['beta']:.3f}; chosen {', '.join(ch['properties']) or 'none'}, "
              f"coefficients {', '.join(f'{c:.3f}' for c in ch['coef'])}; calibration R² {ch['r2']:.2f}): "
              f"RMSE {o['rmse_chosen']:.3f} against {o['rmse_mean']:.3f} for the calibration mean; correlation "
              f"{_f(o['corr'])}.", "", "| held-out set | property | α [95% CI] | predicted |", "|---|---|---|---|"]
        for g, a in o["heldout"].items():
            L.append(f"| {g} | {', '.join(_f(v, 3) for v in o['features'][g].values())} | {_f(a['est'])} "
                     f"[{_f(a['lo'])}, {_f(a['hi'])}] | {_f(o['pred'][g])} |")
        L.append("")
    L += ["## 2. Weaker nulls", "",
          "H6c's gap at E₀ (GP's K₅₀ minus the null's reach, bits_sem; the registered gap and 95% CI shifted by the "
          "difference between the primary null's computed reach and the other null's) and each null's reach per budget "
          "doubling (1 to 16 × E₀) with its price per bit.", "",
          "| domain | null | gap at E₀ [95% CI] | null's reach per doubling | evaluations per bit |", "|---|---|---|---|---|"]
    names = dict(enumeration="exact (primary)", bottom_up="bottom-up, duplicates included",
                 uniform="uniform random search")
    for d, n in res["nulls"].items():
        for k in KINDS:
            v = n[k]
            L.append(f"| {d} | {names[k]} | {_f(v['gap'], sign=True)} [{_f(v['lo'], sign=True)}, "
                     f"{_f(v['hi'], sign=True)}] | {_f(v['rate'])} | ×{_f(v['per_bit'])} |")
    L += ["", "## 3. Tolerance-based reach (real domain)", "",
          "Success = validation nMSE ≤ 10⁻⁶ at the budget (`gen_tol`), beside exact recovery. Boolean: "]
    b = res["tolerance"]["bool"]
    L[-1] += f"tolerance 0, the same outcome as exact in {b['same']} of {b['rows']} runs."
    t = res["tolerance"]["real"]
    L += ["", f"Domain gap at E₀: tolerance {_f(t['gap_tol']['gap'], sign=True)} [{_f(t['gap_tol']['lo'], sign=True)}, "
          f"{_f(t['gap_tol']['hi'], sign=True)}]; exact (registered) {_f(t['gap_exact']['gap'], sign=True)} "
          f"[{_f(t['gap_exact']['lo'], sign=True)}, {_f(t['gap_exact']['hi'], sign=True)}].", "",
          "| set | K₅₀ exact | K₅₀ tolerance | share solved at E₀ exact / tolerance | bits per doubling exact | "
          "bits per doubling tolerance [95% CI] |", "|---|---|---|---|---|---|"]
    for g, v in t["per_set"].items():
        L.append(f"| {g} | {_f(v['k50_exact'])} | {_f(v['k50_tol'])} | {_f(v['share_exact'])} / {_f(v['share_tol'])} | "
                 f"{_f(v['rate_exact']['rate'], 3)} | {_f(v['rate_tol']['rate'], 3)} [{_f(v['rate_tol']['lo'], 3)}, "
                 f"{_f(v['rate_tol']['hi'], 3)}] |")
    bb = res.get("bool_bounds")
    if bb:
        names = dict(registered="registered (uncapped)", capped=f"capped at {bb['space_bits']:.0f} bits",
                     lower_bound="constructed at lower bound, capped", proven_only="proven targets only")
        L += ["", "## 4. Boolean size bounds (guardrail row 154)", "",
              f"A 5-input Boolean set has 2^32 functions, so no size exceeds {bb['space_bits']:.0f} bits_sem; the "
              "registered unit extrapolates above the enumeration ceiling without that bound. The registered reading "
              "reproduces the registered H6c gap exactly (checked). G = GP's K₅₀ at E₀ minus the registered twin's.", "",
              "| reading | gap at E₀ [95% CI] | mean G | sets with K₅₀ above the bound at some budget |", "|---|---|---|---|"]
        for k, v in bb["readings"].items():
            n_above = sum(bool(p["budgets_above_space"]) for p in v["per_set"].values())
            L.append(f"| {names[k]} | {_f(v['gap']['gap'], sign=True)} [{_f(v['gap']['lo'], sign=True)}, "
                     f"{_f(v['gap']['hi'], sign=True)}] | {_f(v['G_mean'], sign=True)} | {n_above} of "
                     f"{len(v['per_set'])} |")
        L += ["", "| set | " + " | ".join(f"K₅₀ / bits per doubling, {k}" for k in bb["readings"]) + " |",
              "|---|" + "---|" * len(bb["readings"])]
        for g in bb["readings"]["registered"]["per_set"]:
            L.append(f"| {g} | " + " | ".join(
                f"{_f(v['per_set'][g]['k50_e0'])} / {_f(v['per_set'][g]['rate']['rate'])}"
                for v in bb["readings"].values()) + " |")
    return "\n".join(L) + "\n"


def main(argv=None, root=None):
    from reach.paths import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    root = Path(root or ROOT)
    res = build(root, a.boot)
    (root / "data/analysis/preempt.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (root / "docs/status/preempt_report.md").write_text(render(res))
    print("written data/analysis/preempt.json and docs/status/preempt_report.md")


if __name__ == "__main__":
    main()

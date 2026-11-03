"""Confirmatory analysis of the full EML-family study (docs/eml_family/preregistration.md, D023).

H1  node invariance (GP-A, E0): every non-EDL operator's exact-recovery K50 lies within +-1.5 nodes of the pooled
    six-operator K50 (equivalence: the 90% target-bootstrap CI of each difference inside +-1.5).
H2  EDL exception: EDL K50 - pooled six < 0 (95% CI upper limit < 0).
H3  prediction: one logistic curve exact ~ K fitted on the confirmatory scout runs (all 7 operators) predicts the
    calculator targets (one and two variables): AUC >= 0.85 and calibration slope in [0.7, 1.3].
H4  budget: at 16x E0 (sizes 9..17) H1's equivalence holds (+-1.5), and every operator's K50 is larger than at
    E0 on the same sizes (95% CI lower limit of the increase > 0).
H4b budget 64x E0 on a subset (EML, LDE0, PLM; sizes 11..17): each within +-1.5 nodes of the pooled three.
H5  algorithm: under the (1+4)-ES at E0 (sizes 3..15), every non-EDL operator within +-2.0 nodes of the pooled six.
Exploratory: PySR 2.7 on the calculator targets (Spearman correlation of exact-recovery rate with minimal K).

    .venv/bin/python -m reach.analysis.family_full [--gp full_gp --es full_es --budget full_budget]
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

from reach.analysis.k50 import logit
from reach.family import FAMILY
from reach.paths import ROOT
from reach.runner.queue import Phase

SIX = [g for g in FAMILY if g != "F-EDL"]
B = 2000
RNG = np.random.default_rng(20261101)


def rows_of(phase, exps, gens=None):
    ph = Phase(phase)
    if not ph.manifest_path.exists():
        return []
    out = []
    for r in ph.results().values():
        if r.get("exp") in exps and (gens is None or r.get("generations") == gens):
            out.append(dict(op=r["grammar"], tid=r["target_id"], K=float(r["K"]),
                            y=float(r.get("exact_hash_match") == "true")))
    return out


def _k50(K, y):
    if len(y) == 0 or y.min() == y.max() or K.min() == K.max():
        return float("nan")
    try:
        b = logit(K, y)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(-b[0] / b[1]) if b[1] < 0 else float("nan")


def k50_boot(rows, ops, levels=None):
    """Point K50 per operator and for the pooled SIX, plus B joint target-bootstrap replicates
    (targets resampled within each operator)."""
    rows = [r for r in rows if levels is None or r["K"] in levels]
    by = {g: defaultdict(list) for g in ops}
    for r in rows:
        if r["op"] in by:
            by[r["op"]][r["tid"]].append((r["K"], r["y"]))
    def est(sample):
        res, allK, ally = {}, [], []
        for g in ops:
            K = np.array([k for t in sample[g] for k, _ in t]); y = np.array([v for t in sample[g] for _, v in t])
            res[g] = _k50(K, y)
            if g in SIX:
                allK.append(K); ally.append(y)
        res["SIX"] = _k50(np.concatenate(allK), np.concatenate(ally)) if allK else float("nan")
        return res
    point = est({g: list(by[g].values()) for g in ops})
    reps = []
    for _ in range(B):
        sample = {}
        for g in ops:
            ts = list(by[g].values())
            sample[g] = [ts[i] for i in RNG.integers(0, len(ts), len(ts))] if ts else []
        reps.append(est(sample))
    return point, reps


def ci(vals, conf):
    v = np.array([x for x in vals if not math.isnan(x)])
    if len(v) < 0.9 * B:
        return float("nan"), float("nan")
    a = (1 - conf) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def equivalence(point, reps, margin, conf=0.90):
    out, ok = [], True
    for g in SIX:
        d = point[g] - point["SIX"]
        lo, hi = ci([r[g] - r["SIX"] for r in reps], conf)
        passed = (not math.isnan(lo)) and -margin < lo and hi < margin
        ok &= passed
        out.append((g, point[g], d, lo, hi, passed))
    return ok, out


def auc(y, p):
    o = np.argsort(p); y = np.asarray(y)[o]; n1 = y.sum(); n0 = len(y) - n1
    r = np.arange(1, len(y) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", default="full_gp")
    ap.add_argument("--es", default="full_es")
    ap.add_argument("--budget", default="full_budget")
    ap.add_argument("--scout_exp", default="conf")
    ap.add_argument("--out", default="docs/results/eml_family_full_report.md")
    a = ap.parse_args()
    L = ["# EML-family full study: confirmatory results", "",
         f"Pre-registration: docs/eml_family/preregistration.md. Phases: {a.gp}, {a.es}, {a.budget}. "
         f"Exact recovery; K50 CIs from {B} joint target-bootstrap replicates.", ""]
    verdict = {}
    gp = rows_of(a.gp, {a.scout_exp})
    if gp:
        point, reps = k50_boot(gp, FAMILY)
        ok, tab = equivalence(point, reps, 1.5)
        verdict["H1"] = ok
        L += ["## H1 node invariance (GP-A, E0; margin ±1.5 nodes, 90% CI)", "",
              f"Pooled six-operator K50 = {point['SIX']:.2f}.", "",
              "| Operator | K50 | difference to pooled six | 90% CI | within ±1.5 |", "|---|---|---|---|---|"]
        L += [f"| {g} | {k:.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'yes' if p else 'NO'} |" for g, k, d, lo, hi, p in tab]
        lo, hi = ci([r["F-EDL"] - r["SIX"] for r in reps], 0.95)
        verdict["H2"] = (not math.isnan(hi)) and hi < 0
        L += ["", f"**H1: {'SUPPORTED' if ok else 'NOT supported'}.**", "",
              "## H2 EDL exception", "", f"EDL K50 = {point['F-EDL']:.2f}; EDL − pooled six = "
              f"{point['F-EDL'] - point['SIX']:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}]. "
              f"**H2: {'SUPPORTED' if verdict['H2'] else 'NOT supported'}.**", ""]
        # H3
        Ks = np.array([r["K"] for r in gp]); ys = np.array([r["y"] for r in gp])
        b = logit(Ks, ys)
        calc = rows_of(a.gp, {"calc", "calc2"})
        if calc:
            Kc = np.array([r["K"] for r in calc]); yc = np.array([r["y"] for r in calc])
            p = 1 / (1 + np.exp(-(b[0] + b[1] * Kc)))
            lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
            try:
                cal = logit(lp, yc)[1]
            except np.linalg.LinAlgError:
                cal = float("nan")
            A = auc(yc, p)
            verdict["H3"] = A >= 0.85 and 0.7 <= cal <= 1.3
            by = defaultdict(list)
            for r, pi in zip(calc, p):
                by[(r["op"], r["tid"].split("_calc_")[1], int(r["K"]))].append((r["y"], pi))
            L += ["## H3 prediction of calculator functions from minimal size", "",
                  f"Curve fitted on {len(gp)} scout runs: K50 = {-b[0] / b[1]:.2f}. Calculator runs: {len(calc)} on "
                  f"{len(by)} (operator, function) targets. **AUC = {A:.3f}, calibration slope = {cal:.2f}. "
                  f"H3: {'SUPPORTED' if verdict['H3'] else 'NOT supported'}.**", "",
                  "| Operator | Function | K | observed | predicted |", "|---|---|---|---|---|"]
            L += [f"| {o} | {f} | {k} | {np.mean([v for v, _ in vv]):.2f} | {np.mean([q for _, q in vv]):.2f} |"
                  for (o, f, k), vv in sorted(by.items(), key=lambda x: (x[0][2], x[0][0]))]
            L.append("")
    bud = rows_of(a.budget, {"budget"}, gens=3200)
    if bud and gp:
        lv = {9.0, 11.0, 13.0, 15.0, 17.0}
        pb, rb = k50_boot(bud, FAMILY)
        p0, r0 = k50_boot(gp, FAMILY, levels=lv)
        ok, tab = equivalence(pb, rb, 1.5)
        inc = []
        for g in FAMILY:
            lo, hi = ci([x[g] - y[g] for x, y in zip(rb, r0)], 0.95)
            inc.append((g, pb[g] - p0[g], lo, hi, (not math.isnan(lo)) and lo > 0))
        verdict["H4"] = ok and all(i[-1] for i in inc)
        L += ["## H4 budget (16× E0, sizes 9–17)", "", "| Operator | K50 at 16× | diff to pooled six [90% CI] | within ±1.5 | increase over E0 [95% CI] | > 0 |",
              "|---|---|---|---|---|---|"]
        incd = {i[0]: i for i in inc}
        for g in FAMILY:
            e = next((t for t in tab if t[0] == g), None)
            eq = f"{e[2]:+.2f} [{e[3]:+.2f}, {e[4]:+.2f}] | {'yes' if e[5] else 'NO'}" if e else "– | –"
            i = incd[g]
            L.append(f"| {g} | {pb[g]:.2f} | {eq} | {i[1]:+.2f} [{i[2]:+.2f}, {i[3]:+.2f}] | {'yes' if i[4] else 'NO'} |")
        L += ["", f"**H4: {'SUPPORTED' if verdict['H4'] else 'NOT supported'}.**", ""]
    b64 = rows_of("full_budget64", {"budget64"}, gens=12800)
    if b64:
        three = ["F-EML", "F-LDE0", "F-PLM"]
        by = defaultdict(lambda: defaultdict(list))
        for r in b64:
            by[r["op"]][r["tid"]].append((r["K"], r["y"]))
        def est(sample):
            res = {g: _k50(np.array([k for t in sample[g] for k, _ in t]), np.array([v for t in sample[g] for _, v in t])) for g in three}
            allt = [x for g in three for t in sample[g] for x in t]
            res["POOL"] = _k50(np.array([k for k, _ in allt]), np.array([v for _, v in allt]))
            return res
        pt = est({g: list(by[g].values()) for g in three})
        reps = []
        for _ in range(B):
            reps.append(est({g: [list(by[g].values())[i] for i in RNG.integers(0, len(by[g]), len(by[g]))] for g in three}))
        rowsH, okb = [], True
        for g in three:
            lo, hi = ci([r[g] - r["POOL"] for r in reps], 0.90)
            ok_ = (not math.isnan(lo)) and -1.5 < lo and hi < 1.5
            okb &= ok_
            rowsH.append(f"| {g} | {pt[g]:.2f} | {pt[g] - pt['POOL']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'yes' if ok_ else 'NO'} |")
        verdict["H4b"] = okb
        L += ["## H4b budget 64× E0 (subset: EML, LDE0, PLM; sizes 11–17; margin ±1.5 around the pooled three)", "",
              f"Pooled K50 = {pt['POOL']:.2f}.", "", "| Operator | K50 | difference | 90% CI | within ±1.5 |", "|---|---|---|---|---|"] + rowsH + \
             ["", f"**H4b: {'SUPPORTED' if okb else 'NOT supported'}.**", ""]
    pys = []
    for g in FAMILY:
        f = ROOT / f"data/family/pysr_{g}.jsonl"
        if f.exists():
            pys += [json.loads(l) for l in open(f) if l.strip()]
    if pys:
        by = defaultdict(list)
        for r in pys:
            by[(r["grammar"], r["function"], r["K"], r["budget"])].append(float(r["exact"]))
        Ks = np.array([k[2] for k in by]); rate = np.array([np.mean(v) for v in by.values()])
        from scipy.stats import spearmanr
        rho = spearmanr(Ks, rate).correlation if len(set(Ks)) > 1 else float("nan")
        L += ["## Exploratory: PySR 2.7 on the calculator targets (constants disabled; IEEE arithmetic)", "",
              f"{len(pys)} runs; Spearman correlation between minimal K and exact-recovery rate across (operator, function, budget) cells: "
              f"rho = {rho:.2f}. Errors recorded: {sum(bool(r['error']) for r in pys)}.", "",
              "| Operator | Function | K | budget | exact rate |", "|---|---|---|---|---|"]
        L += [f"| {g} | {fn} | {k} | {b} | {np.mean(v):.2f} |" for (g, fn, k, b), v in sorted(by.items(), key=lambda x: (x[0][2], x[0][0], x[0][3]))]
        L.append("")
    es = rows_of(a.es, {"es"})
    if es:
        pe, re_ = k50_boot(es, FAMILY)
        ok, tab = equivalence(pe, re_, 2.0)
        verdict["H5"] = ok
        L += ["## H5 second algorithm ((1+4)-ES, E0, sizes 3–15; margin ±2.0)", "", f"Pooled six K50 = {pe['SIX']:.2f}.", "",
              "| Operator | K50 | difference | 90% CI | within ±2.0 |", "|---|---|---|---|---|"]
        L += [f"| {g} | {k:.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'yes' if p else 'NO'} |" for g, k, d, lo, hi, p in tab]
        L += ["", f"**H5: {'SUPPORTED' if ok else 'NOT supported'}.**", ""]
    L += ["## Verdicts", "", " · ".join(f"{h}: {'supported' if v else 'not supported'}" for h, v in verdict.items()) or "no data yet"]
    out = ROOT / a.out
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}; " + ", ".join(f"{h} {v}" for h, v in verdict.items()))


if __name__ == "__main__":
    main()

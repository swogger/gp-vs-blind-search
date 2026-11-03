"""Confirmatory analysis of the held-out extension (OSF addendum; D027). Written before any held-out search run.

H6  prediction: the frozen corrected predictor (data/heldout/predictor.json; per-target predictions in
    data/heldout/predictions.json) predicts exact recovery in every held-out GP run at E0 (random and calculator
    targets, one and two variables): AUC >= 0.85 AND calibration slope in [0.7, 1.3] (slope of a logistic
    regression of the outcome on the predicted logit) -- the bar H3 did not meet.
H7  node invariance replicates: each held-out grammar's K50 (random targets, E0) lies within +-1.5 nodes of the
    frozen six-operator K50 from full_gp, 12.41 (90% target-bootstrap CI of the difference inside +-1.5).
H8  GP without crossover (regime B, `nox_gp`; same targets and seeds as full_gp's scout runs): each of the six
    non-EDL operators' K50 lies within +-1.5 nodes of the pooled six (90% target-bootstrap CI inside the margin),
    the H1 test applied to GP without crossover. Reported with the operator spread (max - min K50) for GP with
    and without crossover.
Outcomes use the corrected exact-match hash (signed-zero fix, 2026-10-15). Error runs count as failures; B = 2000 bootstrap replicates; RNG seed 20261115; a CI needs >= 90% estimable
replicates, otherwise the hypothesis is not supported.

    .venv/bin/python -m reach.analysis.heldout_analysis
"""
import json
import math
from collections import defaultdict

import numpy as np

from reach.analysis.k50 import logit
from reach.paths import ROOT
from reach.runner.phases import HELDOUT
from reach.runner.queue import Phase

B, POOLED_SIX, RNG = 2000, 12.41, np.random.default_rng(20261115)


def rows(phase, exps):
    ph = Phase(phase)
    if not ph.manifest_path.exists():
        return []
    return [dict(g=r["grammar"].replace("S2-", "S-"), tid=r["target_id"], K=float(r["K"]),
                 y=float(r.get("exact_hash_match") == "true")) for r in ph.results().values() if r.get("exp") in exps]


def k50(K, y):
    K, y = np.asarray(K, float), np.asarray(y, float)
    if len(y) == 0 or y.min() == y.max() or K.min() == K.max():
        return float("nan")
    try:
        b = logit(K, y)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(-b[0] / b[1]) if b[1] < 0 else float("nan")


def ci(vals, conf):
    v = np.array([x for x in vals if not math.isnan(x)])
    if len(v) < 0.9 * B:
        return float("nan"), float("nan")
    a = (1 - conf) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def clusters(rs):
    by = defaultdict(lambda: defaultdict(list))
    for r in rs:
        by[r["g"]][r["tid"]].append((r["K"], r["y"]))
    return {g: list(t.values()) for g, t in by.items()}


def k50_of(ts):
    return k50([k for t in ts for k, _ in t], [v for t in ts for _, v in t])


def auc(y, p):
    o = np.argsort(p); y = np.asarray(y)[o]; n1 = y.sum(); n0 = len(y) - n1
    return float((np.arange(1, len(y) + 1)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    L = ["# Held-out extension: confirmatory results (OSF addendum)", "",
         "Analysis: `reach.analysis.heldout_analysis` (registered). Held-out grammars: "
         + ", ".join(f"S-{h}" for h in HELDOUT) + ". Exact recovery; error runs = failures.", ""]
    verdict = {}
    pred = json.loads((ROOT / "data/heldout/predictions.json").read_text())
    gp = rows("heldout_gp", {"conf", "calc", "calc2"})
    if gp:
        y = np.array([r["y"] for r in gp]); p = np.array([pred[r["tid"]]["p"] for r in gp])
        lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
        try:
            cal = logit(lp, y)[1]
        except np.linalg.LinAlgError:
            cal = float("nan")
        A = auc(y, p)
        verdict["H6"] = A >= 0.85 and 0.7 <= cal <= 1.3
        L += ["## H6 prediction on held-out grammars", "",
              f"{len(gp)} runs on {len({r['tid'] for r in gp})} targets. **AUC = {A:.3f}, calibration slope = {cal:.2f}. "
              f"H6: {'SUPPORTED' if verdict['H6'] else 'NOT supported'}.** Mean observed {y.mean():.3f}, mean predicted {p.mean():.3f}.", "",
              "| Grammar | runs | observed | predicted |", "|---|---|---|---|"]
        for h in HELDOUT:
            m = np.array([r["g"] == f"S-{h}" for r in gp])
            L.append(f"| S-{h} | {m.sum()} | {y[m].mean():.3f} | {p[m].mean():.3f} |")
        L.append("")
        # H7
        cl = clusters([r for r in gp if r["tid"] in pred and pred[r["tid"]]["target_set"] == "targets_conf"])
        point = {g: k50_of(ts) for g, ts in cl.items()}
        reps = [{g: k50_of([ts[i] for i in RNG.integers(0, len(ts), len(ts))]) for g, ts in cl.items()} for _ in range(B)]
        ok7 = len(cl) == len(HELDOUT)
        L += ["## H7 node invariance on held-out grammars (margin ±1.5 around the frozen 12.41, 90% CI)", "",
              "| Grammar | K50 | difference | 90% CI | within ±1.5 |", "|---|---|---|---|---|"]
        for g in sorted(cl):
            lo, hi = ci([r[g] - POOLED_SIX for r in reps], 0.90)
            ok = (not math.isnan(lo)) and -1.5 < lo and hi < 1.5
            ok7 &= ok
            L.append(f"| {g} | {point[g]:.2f} | {point[g] - POOLED_SIX:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'yes' if ok else 'NO'} |")
        verdict["H7"] = ok7
        L += ["", f"**H7: {'SUPPORTED' if ok7 else 'NOT supported'}.**", ""]
    nox = [dict(op=r["g"], tid=r["tid"], K=r["K"], y=r["y"]) for r in rows("nox_gp", {"confB"})]
    if nox:
        from reach.analysis import family_full as F
        F.RNG = np.random.default_rng(20261115)
        point, reps = F.k50_boot(nox, F.FAMILY)
        ok8, tab = F.equivalence(point, reps, 1.5)
        verdict["H8"] = ok8
        six = [point[g] for g in F.SIX]
        L += ["## H8 GP without crossover (regime B, E0; margin ±1.5 around the pooled six, 90% CI)", "",
              f"Pooled six K50 = {point['SIX']:.2f}; EDL K50 = {point['F-EDL']:.2f}; spread of the six (max − min) = "
              f"{max(six) - min(six):.2f} nodes.", "",
              "| Operator | K50 | difference | 90% CI | within ±1.5 |", "|---|---|---|---|---|"]
        L += [f"| {g} | {k:.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'yes' if p else 'NO'} |" for g, k, d, lo, hi, p in tab]
        L += ["", f"**H8: {'SUPPORTED' if ok8 else 'NOT supported'}.**", ""]
    L += ["## Verdicts", "", " · ".join(f"{k}: {'supported' if v else 'not supported'}" for k, v in verdict.items()), ""]
    out = ROOT / "docs/results/heldout_report.md"
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}; " + ", ".join(f"{k} {v}" for k, v in verdict.items()))


if __name__ == "__main__":
    main()

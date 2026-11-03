"""Indicative read on: is rigidity predictive beyond scale?

conv ~ n_crit vs conv ~ K, by deviance / AIC / AUC, plus the sharper
within-K test (does alpha separate outcomes at fixed K?).

Runs within a target are not independent -- the honest cluster unit is the
target -- so every model is also fitted at target level (converged out of 20)
and run-level inference is reported with cluster-robust standard errors.
The analyst's version is authoritative; this is a pre-write sanity read.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import statsmodels.api as sm

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "results", "synthetic_dense_k"))


def load(runs_file="synthetic_runs.csv", targets_file="synthetic_targets.csv"):
    tg = {t["target_id"]: t for t in
          csv.DictReader(open(os.path.join(RES, targets_file), encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(open(os.path.join(RES, runs_file), encoding="utf-8")):
        t = tg[r["target_id"]]
        rows.append({
            "target_id": r["target_id"], "regime": r["regime"],
            "y": 1.0 if r["converged"] == "true" else 0.0,
            "K": float(r["K_construction"]),
            "alpha": float(t["alpha_rigidity"]),
            "n_crit": float(t["n_crit"]),
            "g": 1.0 if t["gradient_absence_g"] == "true" else 0.0,
            "rigid": t["rigid_subset"] == "true",
        })
    return rows


def auc(y, p):
    order = np.argsort(p)
    y = np.asarray(y)[order]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.empty(len(p))
    sp = np.sort(p)
    i = 0
    while i < len(sp):                      # average ranks over ties
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit(rows, cols, groups=None):
    X = sm.add_constant(np.array([[r[c] for c in cols] for r in rows]), has_constant="add")
    y = np.array([r["y"] for r in rows])
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    p = m.predict(X)
    out = {"cols": cols, "deviance": m.deviance, "aic": m.aic, "auc": auc(y, p),
           "params": dict(zip(["const"] + cols, m.params))}
    if groups is not None:
        rm = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})
        out["cluster_z"] = dict(zip(["const"] + cols, rm.tvalues))
        out["cluster_p"] = dict(zip(["const"] + cols, rm.pvalues))
    return out


def show(tag, res):
    ps = "  ".join(f"{k}={v:+.4f}" for k, v in res["params"].items() if k != "const")
    line = (f"  {tag:<34} dev={res['deviance']:9.2f}  AIC={res['aic']:9.2f}  "
            f"AUC={res['auc']:.4f}   {ps}")
    if "cluster_p" in res:
        line += "   " + " ".join(
            f"p({k})={v:.2g}" for k, v in res["cluster_p"].items() if k != "const")
    print(line)


def main():
    rows = load()
    print("=" * 100)
    print("RUN-LEVEL MODELS (n=%d runs, %d targets; cluster-robust p by target)"
          % (len(rows), len({r['target_id'] for r in rows})))
    for regime in ("A", "B", "both"):
        rr = rows if regime == "both" else [r for r in rows if r["regime"] == regime]
        gr = [r["target_id"] for r in rr]
        print(f"\n regime {regime}:")
        for cols in (["K"], ["n_crit"], ["K", "alpha"], ["K", "alpha", "g"],
                     ["K", "n_crit"]):
            show(" + ".join(cols), fit(rr, cols, groups=gr))

    print("\n" + "=" * 100)
    print("TARGET-LEVEL (converged out of 20 runs per target, both regimes pooled)")
    agg = {}
    for r in rows:
        a = agg.setdefault(r["target_id"], {"n": 0, "k": 0, **r})
        a["n"] += 1
        a["k"] += r["y"]
    tl = [{"y": v["k"] / v["n"], "K": v["K"], "alpha": v["alpha"],
           "n_crit": v["n_crit"], "g": v["g"], "w": v["n"]} for v in agg.values()]
    X_cols = (["K"], ["n_crit"], ["K", "alpha"])
    for cols in X_cols:
        X = sm.add_constant(np.array([[t[c] for c in cols] for t in tl]), has_constant="add")
        y = np.array([t["y"] for t in tl])
        w = np.array([t["w"] for t in tl])
        m = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit()
        print(f"  {' + '.join(cols):<34} dev={m.deviance:9.2f}  AIC={m.aic:9.2f}  "
              + "  ".join(f"{c}={v:+.4f}" for c, v in zip(cols, m.params[1:])))

    print("\n" + "=" * 100)
    print("WITHIN-K: does rigidity separate outcomes at fixed scale?")
    print("  (pre-registered sample; rigid = alpha>=0.5 AND g)")
    print(f"  {'K':>3}  {'regime':>6}  {'rigid n':>7} {'conv':>6}   "
          f"{'non-rigid n':>11} {'conv':>6}   {'diff':>7}")
    for K in sorted({r["K"] for r in rows}):
        for regime in ("A", "B"):
            rr = [r for r in rows if r["K"] == K and r["regime"] == regime]
            ri = [r for r in rr if r["rigid"]]
            nr = [r for r in rr if not r["rigid"]]
            if not ri or not nr:
                continue
            cr, cn = np.mean([r["y"] for r in ri]), np.mean([r["y"] for r in nr])
            print(f"  {int(K):>3}  {regime:>6}  {len(ri):>7} {cr:>6.3f}   "
                  f"{len(nr):>11} {cn:>6.3f}   {cr-cn:>+7.3f}")

    print("\n" + "=" * 100)
    print("HIGH-K FLOOR CHECK: do non-rigid targets also fail at ~0%?")
    for lo in (19, 21):
        rr = [r for r in rows if r["K"] >= lo]
        for lab, sel in (("rigid", True), ("non-rigid", False)):
            s = [r for r in rr if r["rigid"] == sel]
            print(f"  K>={lo}  {lab:<10} n={len(s):>5}  conv={np.mean([r['y'] for r in s]):.4f}")

    print("\n" + "=" * 100)
    print("LOW-K (K<=11) alpha effect, scale held by K term, cluster-robust:")
    rr = [r for r in rows if r["K"] <= 11]
    show("K + alpha  [K<=11]", fit(rr, ["K", "alpha"], groups=[r["target_id"] for r in rr]))
    show("K + alpha + g  [K<=11]", fit(rr, ["K", "alpha", "g"],
                                       groups=[r["target_id"] for r in rr]))


if __name__ == "__main__":
    main()

"""Do enumeration-only target properties explain GP searchability across EML-family operators?
Exploratory (several candidate predictors, 7 operators). -> docs/results/eml_family_predictors.md"""
import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from reach.family import FAMILY
from reach.paths import ROOT


def load(phase="scout_family", suffix=""):
    pred = pd.concat([pd.DataFrame(json.loads((ROOT / f"data/family/predictors_{g}{suffix}.json").read_text())).assign(op=g)
                      for g in FAMILY])
    runs = [json.loads(l) for l in open(ROOT / f"data/runs/{phase}/results.jsonl")]
    r = pd.DataFrame([dict(target_id=x["target_id"], exact=int(x["exact_hash_match"] == "true"),
                           conv=int(x["converged"] == "true")) for x in runs if x["exp"] == "scout"])
    d = r.merge(pred, on="target_id")
    d["ld"] = np.log10(d["d_min"].clip(lower=1e-20))
    d["near6"] = (d["n_near_6"] > 0).astype(int)
    d["lnear2"] = np.log10(1 + d["n_near_2"])
    return d


def drop_constant(formula, d):
    """Drop predictors with a single value in d (e.g. near6 after the near-duplicate filter)."""
    terms = [s.strip() for s in formula.split("+")]
    keep = [s for s in terms if s.startswith("C(") or s not in d or d[s].nunique() > 1]
    return " + ".join(keep)


def fit(formula, d, y):
    formula = drop_constant(formula, d)
    m = smf.glm(f"{y} ~ {formula}", d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["target_id"])[0]})
    return m


def auc(y, p):
    order = np.argsort(p)
    y = np.asarray(y)[order]
    n1, n0 = y.sum(), len(y) - y.sum()
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logo(d, formula, y):
    """Leave one operator out: mean held-out log-loss and per-operator K50 prediction error."""
    ll, rows = [], []
    for g in FAMILY:
        tr, te = d[d.op != g], d[d.op == g]
        m = smf.glm(f"{y} ~ {drop_constant(formula, d)}", tr, family=sm.families.Binomial()).fit()
        p = np.clip(m.predict(te), 1e-9, 1 - 1e-9)
        ll.append(float(-np.mean(te[y] * np.log(p) + (1 - te[y]) * np.log(1 - p))))
        rows.append((g, float(te[y].mean()), float(p.mean())))
    return float(np.mean(ll)), rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="scout_family")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="docs/results/eml_family_predictors.md")
    a = ap.parse_args()
    d = load(a.phase, a.suffix)
    loc = all(c in d for c in ("nb_med", "nb_nonfinite", "nb_good"))
    L = ["# EML-family: do enumeration-only properties predict searchability?", "",
         f"Exploratory analysis: {len(d)} scout runs on {d.target_id.nunique()} targets (K ≤ 19) across {d.op.nunique()} "
         "operators; outcome = exact recovery at E₀ (primary) and convergence. Logistic models with cluster-robust SEs by target. "
         "Predictors come from the enumeration alone: `ld` = log10 of the nMSE of the nearest smaller function (deceptive "
         "near-solution distance), `near6` = some smaller function within nMSE ≤ 10⁻⁶, `lnear2` = log10(1 + number "
         "within 10⁻²), `sub_viable` = share of finite subtrees, `dyn_range` = log10 magnitude ratio.", "",
         "## Target properties by operator (medians; share with a near-duplicate)", "",
         "| Operator | Targets | median ld | share near6 | median lnear2 | mean sub_viable | median dyn_range | exact rate | conv rate |"
         + (" median nb_med | mean nb_nonfinite | mean nb_good |" if loc else ""),
         "|---|---|---|---|---|---|---|---|---|" + ("---|---|---|" if loc else "")]
    for g in FAMILY:
        x = d[d.op == g]
        t = x.drop_duplicates("target_id")
        L.append(f"| {g} | {len(t)} | {t.ld.median():.1f} | {t.near6.mean():.2f} | {t.lnear2.median():.2f} | "
                 f"{t.sub_viable.mean():.2f} | {t.dyn_range.median():.2f} | {x.exact.mean():.2f} | {x.conv.mean():.2f} |"
                 + (f" {t.nb_med.median():.2f} | {t.nb_nonfinite.mean():.2f} | {t.nb_good.mean():.2f} |" if loc else ""))
    for y in ("exact", "conv"):
        L += ["", f"## Models for `{y}`", "", "| Model | AIC | AUC | Operator dummies: LR χ² (df 6), p | Key coefficients (cluster-robust z) |",
              "|---|---|---|---|---|"]
        specs = {"M0 size": "K", "M1 size + operator": "K + C(op)",
                 "M2 size + ld": "K + ld", "M3 size + ld + near6 + sub_viable + dyn_range": "K + ld + near6 + sub_viable + dyn_range",
                 "M4 = M3 + operator": "K + ld + near6 + sub_viable + dyn_range + C(op)"}
        if loc:
            specs["M5 size + locality"] = "K + nb_med + nb_nonfinite + nb_good"
            specs["M6 = M3 + locality"] = "K + ld + near6 + sub_viable + dyn_range + nb_med + nb_nonfinite + nb_good"
            specs["M7 = M6 + operator"] = specs["M6 = M3 + locality"] + " + C(op)"
        fits = {k: fit(v, d, y) for k, v in specs.items()}
        for k, m in fits.items():
            lr = ""
            base = {"M1 size + operator": "M0 size", "M4 = M3 + operator": "M3 size + ld + near6 + sub_viable + dyn_range",
                    "M7 = M6 + operator": "M6 = M3 + locality"}.get(k)
            if base:
                stat = 2 * (m.llf - fits[base].llf)
                from scipy.stats import chi2
                lr = f"{stat:.1f}, p = {chi2.sf(stat, 6):.2g}"
            coefs = ", ".join(f"{n} {m.params[n]:+.2f} (z {m.tvalues[n]:+.1f})" for n in m.params.index
                              if n in ("K", "ld", "near6", "sub_viable", "dyn_range", "nb_med", "nb_nonfinite", "nb_good"))
            L.append(f"| {k} | {m.aic:.0f} | {auc(d[y], m.predict(d)):.3f} | {lr} | {coefs} |")
        L += ["", f"Leave-one-operator-out (`{y}`): mean held-out log-loss, and observed against predicted mean rate per held-out operator.", "",
              "| Model | held-out log-loss | per operator: observed → predicted |", "|---|---|---|"]
        for k in [m for m in ("M0 size", "M2 size + ld", "M3 size + ld + near6 + sub_viable + dyn_range",
                              "M5 size + locality", "M6 = M3 + locality") if m in specs]:
            ll, rows = logo(d, specs[k], y)
            L.append(f"| {k} | {ll:.4f} | " + ", ".join(f"{g[2:]} {o:.2f}→{p:.2f}" for g, o, p in rows) + " |")
    out = ROOT / a.out
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

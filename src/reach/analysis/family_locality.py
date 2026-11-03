"""EXPLORATORY (after the confirmatory runs, D026): do enumeration-only target properties explain the operator
differences under the (1+4)-ES, where node reach is not invariant (H5 not supported), and do they matter for GP?

Metrics per confirmatory target (`targets_conf`) from `reach.family_metrics.target_metrics` without the
near-duplicate scan (those targets passed the near-duplicate filter): locality (one-edit neighbours),
sub_viable, dyn_range. Cached in data/family/locality_conf_<op>.json. Outcome: exact recovery at E0; error
runs count as failures, as in the registered analysis. -> docs/results/eml_family_locality_exploratory.md
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2

from reach import grammar as G
from reach.analysis.family_predictors import auc, fit
from reach.family import FAMILY
from reach.family_metrics import target_metrics
from reach.paths import ROOT

LOC = "nb_med + nb_nonfinite + nb_good + sub_viable + dyn_range"


def metrics(g):
    path = ROOT / f"data/family/locality_conf_{g}.json"
    if not path.exists():
        gr = G.get(g)
        ts = json.loads((ROOT / f"data/targets/{g}/targets_conf.json").read_text())
        out = [dict(target_id=t["target_id"], **target_metrics(gr, G.parse_expr(t["expression"], gr), t["var_target"], {}))
               for t in ts]
        path.write_text(json.dumps(out, indent=1))
    return pd.DataFrame(json.loads(path.read_text())).assign(op=g)


def runs(phase, exp):
    rows = [json.loads(l) for l in open(ROOT / f"data/runs/{phase}/results.jsonl")]
    return pd.DataFrame([dict(target_id=r["target_id"], exact=int(r.get("exact_hash_match") == "true"))
                         for r in rows if r.get("exp") == exp])


def logo(d, formula):
    rows, ll = [], []
    for g in FAMILY:
        tr, te = d[d.op != g], d[d.op == g]
        m = fit(formula, tr, "exact")
        p = np.clip(m.predict(te), 1e-9, 1 - 1e-9)
        ll.append(float(-np.mean(te.exact * np.log(p) + (1 - te.exact) * np.log(1 - p))))
        rows.append((g, te.exact.mean(), p.mean()))
    return float(np.mean(ll)), rows


def main():
    pred = pd.concat([metrics(g) for g in FAMILY])
    L = ["# EXPLORATORY: what explains operator differences under the ES (and under GP)?", "",
         "Not pre-registered; run after the confirmatory phases. Targets: `targets_conf`. Outcome: exact recovery at E₀ "
         "(error runs = failures). Logistic models, cluster-robust SEs by target. Locality = fitness of one-edit "
         "neighbours: `nb_med` median log₁₀ nMSE, `nb_nonfinite` share non-finite, `nb_good` share within nMSE ≤ 10⁻²; "
         "plus `sub_viable`, `dyn_range` (see `reach.family_metrics`).", ""]
    for name, phase, exp, kmax in (("GP", "full_gp", "conf", 19), ("ES", "full_es", "es", 15)):
        d = runs(phase, exp).merge(pred, on="target_id")
        d = d[d.K <= kmax]
        t = d.drop_duplicates("target_id")
        L += [f"## {name} ({len(d)} runs, {t.target_id.nunique()} targets, K ≤ {kmax})", "",
              "| Operator | exact rate | median nb_med | mean nb_nonfinite | mean nb_good | mean sub_viable | median dyn_range |",
              "|---|---|---|---|---|---|---|"]
        for g in FAMILY:
            x, tx = d[d.op == g], t[t.op == g]
            L.append(f"| {g} | {x.exact.mean():.2f} | {tx.nb_med.median():.2f} | {tx.nb_nonfinite.mean():.2f} | "
                     f"{tx.nb_good.mean():.3f} | {tx.sub_viable.mean():.2f} | {tx.dyn_range.median():.2f} |")
        specs = {"size": "K", "size + operator": "K + C(op)", "size + locality": f"K + {LOC}",
                 "size + locality + operator": f"K + {LOC} + C(op)"}
        fits = {k: fit(v, d, "exact") for k, v in specs.items()}
        L += ["", "| Model | AIC | AUC | operator LR χ² (df 6), p | coefficients (cluster-robust z) |", "|---|---|---|---|---|"]
        for k, m in fits.items():
            base = {"size + operator": "size", "size + locality + operator": "size + locality"}.get(k)
            lr = ""
            if base:
                s = 2 * (m.llf - fits[base].llf)
                lr = f"{s:.1f}, p = {chi2.sf(s, 6):.2g}"
            co = ", ".join(f"{n} {m.params[n]:+.2f} (z {m.tvalues[n]:+.1f})" for n in m.params.index
                           if not n.startswith("C(") and n != "Intercept")
            L.append(f"| {k} | {m.aic:.0f} | {auc(d.exact, m.predict(d)):.3f} | {lr} | {co} |")
        L += ["", "Leave-one-operator-out: held-out log-loss; observed → predicted exact rate per held-out operator.", "",
              "| Model | log-loss | per operator |", "|---|---|---|"]
        for k in ("size", "size + locality"):
            ll, rows = logo(d, specs[k])
            L.append(f"| {k} | {ll:.4f} | " + ", ".join(f"{g[2:]} {o:.2f}→{p:.2f}" for g, o, p in rows) + " |")
        L.append("")
    out = ROOT / "docs/results/eml_family_locality_exploratory.md"
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""Held-out extension A4 (D027): choose and freeze the corrected predictor of exact recovery, using the
confirmatory `full_gp` data only (all runs: random targets, one- and two-variable calculator targets).

Candidate models use enumeration-only features: K (minimal size), nvar (1 = two variables), calc (1 = a
calculator function rather than a random minimal target) and nb_good (share of one-edit neighbours within
nMSE <= 1e-2). Selection rule, fixed in advance: smallest leave-one-operator-out log-loss (an operator's
one- and two-variable grammars held out together). The chosen model is refitted on all data and frozen
in data/heldout/predictor.json. -> docs/results/heldout_predictor_selection.md
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from reach import grammar as G
from reach.family import FAMILY
from reach.family_metrics import target_metrics
from reach.paths import ROOT
from reach.runner.queue import Phase

MODELS = {"M0 K (registered H3 form)": "K", "M1 K + nvar": "K + nvar", "M2 K + nvar + calc": "K + nvar + calc",
          "M3 K * calc + nvar": "K * calc + nvar", "M4 K + nvar + nb_good": "K + nvar + nb_good",
          "M5 K * calc + nvar + nb_good": "K * calc + nvar + nb_good"}


def locality(g, tname):
    path = ROOT / f"data/family/locality_{tname}_{g}.json"
    if not path.exists():
        gr = G.get(g)
        ts = json.loads((ROOT / f"data/targets/{g}/{tname}.json").read_text())
        path.write_text(json.dumps([dict(target_id=t["target_id"], **target_metrics(
            gr, G.parse_expr(t["expression"], gr), t["var_target"], {})) for t in ts], indent=1))
    return {r["target_id"]: r for r in json.loads(path.read_text())}


def load(phase="full_gp"):
    """Training rows; outcomes corrected for the signed-zero hash bug (data/rescored/<phase>.json)."""
    flips = set(json.loads((ROOT / f"data/rescored/{phase}.json").read_text()))
    loc = {}
    for g in FAMILY:
        loc.update(locality(g, "conf")); loc.update(locality(g, "calc")); loc.update(locality("F2" + g[1:], "calc2"))
    rows = []
    for k, r in Phase(phase).results().items():
        rows.append(dict(op=r["grammar"].replace("F2-", "F-"), tid=r["target_id"], K=float(r["K"]),
                         nvar=int(r["grammar"].startswith("F2-")), calc=int(r["exp"] in ("calc", "calc2")),
                         nb_good=loc[r["target_id"]]["nb_good"],
                         exact=int(r.get("exact_hash_match") == "true" or k in flips)))
    return pd.DataFrame(rows)


def fit(formula, d):
    return smf.glm(f"exact ~ {formula}", d, family=sm.families.Binomial()).fit()


def scores(y, p):
    p = np.clip(np.asarray(p), 1e-9, 1 - 1e-9); y = np.asarray(y)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    o = np.argsort(p); ys = y[o]; n1 = ys.sum(); n0 = len(ys) - n1
    auc = float((np.arange(1, len(ys) + 1)[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    lp = np.log(p / (1 - p))
    cal = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params
    return ll, auc, float(cal[1]), float(cal[0])


def main():
    d = load()
    L = ["# Held-out extension: choice of the corrected predictor (A4, D027)", "",
         f"Training data: `full_gp` (outcomes corrected for the signed-zero hash bug), {len(d)} runs on {d.tid.nunique()} targets (7 operators; random targets, one- and "
         "two-variable calculator functions). Selection rule fixed in advance: smallest leave-one-operator-out (LOGO) "
         "log-loss. LOGO AUC and calibration (slope, intercept) are pooled over the held-out predictions.", "",
         "| Model | LOGO log-loss | LOGO AUC | LOGO calibration slope | LOGO intercept | in-sample AIC |", "|---|---|---|---|---|---|"]
    best = None
    for name, f in MODELS.items():
        pred = pd.Series(index=d.index, dtype=float)
        for g in FAMILY:
            te = d.op == g
            pred[te] = fit(f, d[~te]).predict(d[te])
        ll, auc, slope, icpt = scores(d.exact, pred)
        L.append(f"| {name} | {ll:.4f} | {auc:.3f} | {slope:.2f} | {icpt:+.2f} | {fit(f, d).aic:.0f} |")
        if best is None or ll < best[1]:
            best = (name, ll)
    name = best[0]
    m = fit(MODELS[name], d)
    L += ["", f"**Chosen: {name}** (`exact ~ {MODELS[name]}`), refitted on all training runs:", "",
          "| Term | Coefficient | SE |", "|---|---|---|"]
    L += [f"| {k} | {v:+.4f} | {m.bse[k]:.4f} |" for k, v in m.params.items()]
    out = ROOT / "data/heldout"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictor.json").write_text(json.dumps(dict(model=name, formula=MODELS[name],
                                                         params={k: float(v) for k, v in m.params.items()},
                                                         training_runs=len(d)), indent=1))
    (ROOT / "docs/results/heldout_predictor_selection.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

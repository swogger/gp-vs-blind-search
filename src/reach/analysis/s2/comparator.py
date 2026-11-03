"""H7 comparator (design H7; D070): a problem-feature model in the style of Graff, Poli and Flores (2013, Evolutionary
Computation 21(4):533-560, doi 10.1162/EVCO_a_00096), whose difficulty indicators are "essentially based on the notion
of finite difference" (checked against the abstract, 2026-10-25). It sees only the target function, never the operator
set: real targets by finite differences on the grid, Boolean targets by sensitivity (the Boolean finite difference),
nonlinearity, degree, essential inputs and imbalance. Fitted on calibration data per domain (logistic, standardised
features, plus log2 budget) and scored on held-out data exactly like the frozen size model.
"""
import math

import numpy as np
import statsmodels.api as sm

from reach import grammar as G
from reach.s2_targets import boolean_properties

REAL_FEATURES = ("mean_abs_d1", "mean_abs_d2", "spread", "turning_points")
BOOL_FEATURES = ("avg_sensitivity", "nonlinearity", "degree", "essential_inputs", "imbalance")


def features(g, expression, scale=None):
    tree = G.parse_expr(expression, g)
    if g.kind == "bool":
        n = g.n_inputs
        tt = G.evaluate(g, tree)
        rows = 1 << n
        f = [(tt >> r) & 1 for r in range(rows)]
        sens = sum(f[r] != f[r ^ (1 << i)] for r in range(rows) for i in range(n)) / rows
        p = boolean_properties(tt, n)
        return dict(avg_sensitivity=float(sens), nonlinearity=float(p["nonlinearity"]), degree=float(p["degree"]),
                    essential_inputs=float(p["essential_inputs"]), imbalance=abs(p["weight"] / rows - 0.5))
    v = G.evaluate(g, tree, G.sample_grid(g)).real
    s = scale or max(1.0, float(np.abs(v).max()))
    d1, d2 = np.diff(v) / s, np.diff(v, 2) / s
    return dict(mean_abs_d1=float(np.mean(np.abs(d1))), mean_abs_d2=float(np.mean(np.abs(d2))),
                spread=float(np.std(v) / s), turning_points=float(np.sum(np.diff(np.sign(d1)) != 0)))


def _X(recs, names, mu, sd):
    Z = np.array([[(r["pfeatures"][k] - mu[i]) / sd[i] for i, k in enumerate(names)] for r in recs], float)
    lb = np.array([math.log2(r["budget"]) for r in recs], float)
    return sm.add_constant(np.column_stack([Z, lb]), has_constant="add")


def fit(recs, names):
    F = np.array([[r["pfeatures"][k] for k in names] for r in recs], float)
    mu, sd = F.mean(axis=0), F.std(axis=0)
    sd[sd == 0] = 1.0
    y = np.array([r["y"] for r in recs], float)
    params = sm.GLM(y, _X(recs, names, mu, sd), family=sm.families.Binomial()).fit().params
    return dict(kind="graff_poli_flores_2013_style", names=list(names), mu=mu.tolist(), sd=sd.tolist(),
                params=[float(x) for x in params])


def predict(model, recs):
    eta = _X(recs, model["names"], model["mu"], model["sd"]) @ np.array(model["params"])
    return 1 / (1 + np.exp(-eta))


def loso_logloss(recs, names):
    """Mean leave-one-set-out log-loss, scored exactly like the size model's selection (predictor.select)."""
    sets = sorted({r["set"] for r in recs})
    out = []
    for s in sets:
        m = fit([r for r in recs if r["set"] != s], names)
        te = [r for r in recs if r["set"] == s]
        p = np.clip(predict(m, te), 1e-9, 1 - 1e-9)
        y = np.array([r["y"] for r in te], float)
        out.append(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()))
    return float(np.mean(out))

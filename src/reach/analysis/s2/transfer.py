"""Transfer of the frozen H7 law to another engine (H10b for CGP, D076; H9b for Operon, D090).

Per engine (and domain or configuration): logit P = c0 + c1 * eta, with eta the frozen H7 model's linear predictor,
fitted by maximum likelihood on the engine's calibration runs at the registered budgets. The fit is a model of the
frozen form (c0 + c1 * a, c1 * b, c1 * g, ...), so `predictor.predict` reads it unchanged. Its Brier baseline (D090:
nodes plus the log2 budget term, as the model; nodes only when the engine runs at one budget, where log2 budget is a
constant) is fitted on the same records and frozen with it. Separated data (`k50.separated` on eta) have no finite c0
and c1 and are refused (guardrail row 118, D088). A frozen file is checked for content and never replaced with other
content (guardrail row 91)."""
import json
import math
from pathlib import Path

import numpy as np
import statsmodels.api as sm

from reach.analysis.s2 import predictor

UNITS = ("nodes", "bits_uniform", "bits_syn", "bits_sem")       # = confirm.UNITS


def linear_predictor(model, recs):
    return predictor._X(recs, model["unit"], model["budget_term"], model.get("feature")) @ np.array(model["params"])


def fit(frozen, recs, budgets, label, baseline_budget_term):
    """c0 + c1 * eta by maximum likelihood on the records at `budgets`, with its frozen baseline (D090)."""
    from reach.analysis.k50 import separated
    recs = [r for r in recs if r["budget"] in budgets]
    if not recs:
        raise ValueError(f"{label}: no records at the fit budgets {list(budgets)}")
    eta = linear_predictor(frozen, recs)
    y = np.array([r["y"] for r in recs], float)
    if separated(eta, y):
        raise ValueError(f"{label}: outcomes separated on the frozen model's eta; no finite c0, c1 (row 118)")
    c0, c1 = (float(x) for x in sm.GLM(y, sm.add_constant(eta, has_constant="add"),
                                       family=sm.families.Binomial()).fit().params)
    p = list(frozen["params"])
    return dict(unit=frozen["unit"], budget_term=frozen["budget_term"], feature=frozen.get("feature"),
                params=[c0 + c1 * p[0]] + [c1 * x for x in p[1:]], transfer=dict(c0=c0, c1=c1),
                base_params=p, base_sha256=frozen.get("sha256"), budgets=list(budgets), n=len(recs),
                baseline=predictor.fit_baseline(recs, budget_term=baseline_budget_term),
                rule=f"logit P = c0 + c1 * eta (frozen H7 model), maximum likelihood on {label} calibration runs, "
                     f"budgets {list(budgets)} x E0")


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_baseline(b, budget_term, label):
    """A frozen baseline (D090): nodes, the stated budget term, no feature, finite coefficients of the right number."""
    if not isinstance(b, dict) or b.get("unit") != "nodes" or b.get("budget_term") is not budget_term \
            or b.get("feature") is not None:
        raise ValueError(f"{label}: baseline must be nodes with budget term {budget_term} (D090), got {b!r}")
    p = b.get("params")
    if not (isinstance(p, list) and len(p) == 2 + budget_term and all(map(_finite, p))):
        raise ValueError(f"{label}: baseline params must be {2 + budget_term} finite numbers")


def check_one(m, budgets, label, baseline_budget_term):
    """One transfer model: a known unit, the fit budgets, finite c0 and c1, params = [c0 + c1 * b0, c1 * b1, ...] of
    its base params, and a valid frozen baseline (D090)."""
    if not isinstance(m, dict):
        raise ValueError(f"{label}: not a model")
    if m.get("unit") not in UNITS or not isinstance(m.get("budget_term"), bool):
        raise ValueError(f"{label}: unit {m.get('unit')!r} / budget term {m.get('budget_term')!r}")
    if m.get("feature") is not None and not isinstance(m.get("feature"), str):
        raise ValueError(f"{label}: feature {m.get('feature')!r}")
    if m.get("budgets") != list(budgets):
        raise ValueError(f"{label}: fit budgets {m.get('budgets')!r}, registered {list(budgets)}")
    n = 2 + m["budget_term"] + (m.get("feature") is not None)
    p, base, t = m.get("params"), m.get("base_params"), m.get("transfer")
    if not (isinstance(p, list) and isinstance(base, list) and len(p) == len(base) == n
            and all(map(_finite, p + base))):
        raise ValueError(f"{label}: params and base_params must be {n} finite numbers")
    if not (isinstance(t, dict) and _finite(t.get("c0")) and _finite(t.get("c1"))):
        raise ValueError(f"{label}: transfer needs finite c0 and c1")
    want = [t["c0"] + t["c1"] * base[0]] + [t["c1"] * x for x in base[1:]]
    if not all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(p, want)):
        raise ValueError(f"{label}: params are not c0 + c1 * eta of its base params")
    check_baseline(m.get("baseline"), baseline_budget_term, label)


def check(models, keys, budgets, label, baseline_budget_term):
    if not isinstance(models, dict) or set(models) != set(keys):
        raise ValueError(f"{label}: keys {sorted(map(str, models)) if isinstance(models, dict) else models!r}, "
                         f"expected exactly {list(keys)}")
    for k in keys:
        check_one(models[k], budgets, f"{label} ({k})", baseline_budget_term)


def freeze(models, path, checker):
    """One file with its .sha256 (predictor.freeze, which never replaces other content); `checker` validates first."""
    checker(models)
    path = Path(path)
    if path.exists() and path.read_text() != json.dumps(models, indent=1, sort_keys=True):     # predictor.freeze's body
        raise RuntimeError(f"{path.name} exists with other content: a frozen transfer model is never replaced. If it "
                           "must change, record a decision and an OSF update, and remove it by hand.")
    return predictor.freeze(models, path)

"""H7: select, freeze and score the predictor of GP success (tool T16; docs/design/hypotheses.md, H7; brief §9.7).

Candidates: logit P = a + b*s_u (+ g*log2 budget), for each unit u. Selection rule (registered): the lowest mean
leave-one-set-out log-loss on calibration data. The frozen model is a JSON file with its SHA-256 (cited in the Stage 2
registration). Scoring on held-out data: AUC; calibration slope and intercept (logistic regression of y on the frozen
model's logit; intercept CI by target bootstrap); Brier score against a nodes-only baseline fitted on calibration
data; per-set absolute error between mean predicted and observed recovery. Passes if AUC >= 0.80, the slope is in
[0.7, 1.3] with an intercept CI covering 0, the Brier score beats the baseline, and at least `min_sets_ok` sets' errors are <= eps_set (registered: 5 of 6;
`score_model` applies the registered values, H7_RULE and SCORED; guardrail row 125)."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import rankdata


def _X(recs, unit, budget_term, feature=None):
    """Columns: 1, s_u, [log2 budget], [one set-level feature, from r["features"]] (design 9.7)."""
    s = np.array([r["units"][unit] for r in recs], float)
    cols = [s] + ([np.array([math.log2(r["budget"]) for r in recs], float)] if budget_term else [])
    cols += [np.array([r["features"][feature] for r in recs], float)] if feature else []
    return sm.add_constant(np.column_stack(cols), has_constant="add")


def _fit(recs, unit, budget_term, feature=None):
    """Maximum-likelihood coefficients; ValueError when separation leaves one without a finite value (row 118)."""
    from reach.analysis.k50 import separated_columns
    y = np.array([r["y"] for r in recs], float)
    X = _X(recs, unit, budget_term, feature)
    if separated_columns(X, y):
        raise ValueError(f"separated data: no maximum-likelihood fit for {unit}, budget term {budget_term}, {feature}")
    return sm.GLM(y, X, family=sm.families.Binomial()).fit().params


def _loso(by, s, u, bt, f):
    try:
        params = _fit([r for o in by if o != s for r in by[o]], u, bt, f)
    except ValueError:
        return float("inf")                     # no maximum-likelihood fit: the candidate cannot be selected (row 118)
    return _logloss(params, by[s], u, bt, f)


def _logloss(params, recs, unit, budget_term, feature=None):
    eta = _X(recs, unit, budget_term, feature) @ params
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-9, 1 - 1e-9)
    y = np.array([r["y"] for r in recs], float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def select(recs, units, budget_terms=(False, True), features=()):
    """Registered rule (design 9.7): among unit x budget term x (no feature or one set-level feature: number of
    variables or one grammar property), the lowest mean leave-one-set-out log-loss. A set-level feature is constant
    within a set, so its coefficient is learned across sets and predicts a new set's baseline. A candidate whose
    leave-one-set-out fit is separated (no maximum-likelihood fit) has loss inf and is never selected."""
    by = defaultdict(list)
    for r in recs:
        by[r["set"]].append(r)
    best = None
    for u in units:
        for bt in budget_terms:
            for f in (None,) + tuple(features):
                loss = np.mean([_loso(by, s, u, bt, f) for s in by])
                if best is None or loss < best["loso_logloss"]:
                    best = dict(unit=u, budget_term=bt, feature=f, loso_logloss=float(loss))
    best["params"] = [float(x) for x in _fit(recs, best["unit"], best["budget_term"], best["feature"])]
    return best


def freeze(model, path):
    """Write the model with its .sha256. An existing file with other content is never replaced (guardrail row 123): a
    frozen model changes only by a recorded decision and an OSF update, and is then removed by hand."""
    path = Path(path)
    body = json.dumps(model, indent=1, sort_keys=True)
    if path.exists() and path.read_text() != body:
        raise RuntimeError(f"{path.name} exists with other content: a frozen model is never replaced. If it must "
                           "change, record a decision and an OSF update, and remove it by hand.")
    path.write_text(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{sha}  {path.name}\n")
    return dict(model, sha256=sha)


def predict(model, recs):
    eta = _X(recs, model["unit"], model["budget_term"], model.get("feature")) @ np.array(model["params"])
    return 1 / (1 + np.exp(-eta))


def fit_baseline(recs, budget_term):
    """The grammar-agnostic baseline for the Brier criterion: logit P = a + b * nodes (+ g * log2 budget). D090 ("Nodes
    + budget"): the budget term is included as in the model, so the test is of the size unit, not of the budget term;
    without it only where the records hold one budget (Operon, 16 x E0), where log2 budget is a constant (refused)."""
    if budget_term and len({r["budget"] for r in recs}) < 2:
        raise ValueError("the baseline's budget term needs records at more than one budget (one budget: a constant)")
    return dict(unit="nodes", budget_term=bool(budget_term), feature=None,
                params=[float(x) for x in _fit(recs, "nodes", bool(budget_term))])


def _auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _calibration(y, p):
    """(intercept, slope) of y on the model's logit; NaN when degenerate (one outcome, or one predicted value: no
    maximum-likelihood fit; REQ-ANA-02, guardrail row 140) or separated (row 118). In the bootstrap such a replicate
    is dropped and counted (`score_p`)."""
    from reach.analysis.k50 import separated
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    if len(np.unique(y)) < 2 or len(np.unique(lp)) < 2 or separated(lp, y):
        return float("nan"), float("nan")
    fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    return float(fit.params[0]), float(fit.params[1])


# The registered H7 criteria (Stage 2 form; B as filed in Stage 1, confirm.REGISTERED) and, per scored test, the
# number of sets per domain, how many must meet the per-set criterion, and the budgets of the scored records. The
# scoring functions below have no default for these values (guardrail row 125); `score_model` applies them.
H7_RULE = dict(auc=0.80, slope=(0.7, 1.3), eps_set=0.10, boot=2000, seed=20261020)
SCORED = {
    "H7": dict(n_sets=6, min_sets_ok=5, budgets=(1, 2, 4, 8, 16)),       # tree GP, regime A, s2_ho_a0 and s2_ho_a16
    "H10b": dict(n_sets=6, min_sets_ok=5, budgets=(1, 2, 4, 8, 16)),     # CGP, s2_ho_cgp
    "H9b": dict(n_sets=2, min_sets_ok=2, budgets=(16,)),                 # Operon per configuration, s2_ho_operon (D090)
}


def score_registered(test, recs, p, pb):
    """The registered scoring of one domain's held-out records for `test` (a key of SCORED): the H7 criteria with
    H7_RULE's values. Records at another budget or more sets than registered are refused; a registered set without
    records counts as failing the per-set criterion (`sets_missing`)."""
    spec = SCORED[test]
    bad = sorted({r["budget"] for r in recs} - set(spec["budgets"]))
    if bad:
        raise ValueError(f"{test}: records at budget(s) {bad}; registered budgets {list(spec['budgets'])}")
    sets = {r["set"] for r in recs}
    if len(sets) > spec["n_sets"]:
        raise ValueError(f"{test}: {len(sets)} sets in the records; {spec['n_sets']} registered per domain")
    res = score_p(recs, p, pb, eps_set=H7_RULE["eps_set"], boot=H7_RULE["boot"], seed=H7_RULE["seed"],
                  min_sets_ok=spec["min_sets_ok"])
    return dict(res, test=test, n_sets_registered=spec["n_sets"], sets_missing=spec["n_sets"] - len(sets),
                criteria=dict(H7_RULE, slope=list(H7_RULE["slope"]), **{k: v for k, v in spec.items() if k != "budgets"},
                              budgets=list(spec["budgets"])))


def score_model(test, model, recs, baseline):
    """The registered entry point: a frozen model (or a transfer model of the frozen form) and its frozen baseline."""
    return score_registered(test, recs, predict(model, recs), predict(baseline, recs))


def describe(recs, p, pb):
    """Benchmarks (D090, "Descriptive, pooled"): per domain, the H7 quantities (AUC, calibration slope and intercept
    with the target-bootstrap CI, Brier against the baseline) on all benchmark cells pooled, and the absolute error
    between mean predicted and observed recovery per cell (set and benchmark) and per set. No pass or fail."""
    res = score_p(recs, p, pb, eps_set=H7_RULE["eps_set"], boot=H7_RULE["boot"], seed=H7_RULE["seed"], min_sets_ok=0)
    for k in ("passes", "sets_ok"):
        res.pop(k)
    p, y = np.asarray(p, float), np.array([r["y"] for r in recs], float)
    cells = defaultdict(list)
    for i, r in enumerate(recs):
        cells[f"{r['set']}|{r['target_id']}"].append(i)
    res["per_cell_error"] = {c: float(abs(p[ix].mean() - y[ix].mean())) for c, ix in sorted(cells.items())}
    return dict(res, descriptive=True, rule="benchmarks: descriptive, pooled per domain, no pass or fail (D090)")


def describe_model(model, recs, baseline):
    return describe(recs, predict(model, recs), predict(baseline, recs))


def score(model, recs, baseline, eps_set, *, boot, min_sets_ok, seed=20261020):
    return score_p(recs, predict(model, recs), predict(baseline, recs), eps_set, boot=boot, seed=seed,
                   min_sets_ok=min_sets_ok)


def score_p(recs, p, pb, eps_set, *, boot, min_sets_ok, seed=20261020):
    """The H7 criteria on given predictions p (and the baseline's pb) for the records: used for the frozen model
    (`score`) and for the comparators, so that all are scored by the same code (D084). eps_set, boot and min_sets_ok
    have no default (row 125): the registered values come from `score_registered`."""
    y = np.array([r["y"] for r in recs], float)
    p, pb = np.asarray(p, float), np.asarray(pb, float)
    icpt, slope = _calibration(y, p)
    by_t = defaultdict(list)
    for i, r in enumerate(recs):
        by_t[r["target_id"]].append(i)
    tids = list(by_t)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(boot):
        idx = np.concatenate([by_t[tids[j]] for j in rng.integers(0, len(tids), len(tids))])
        reps.append(_calibration(y[idx], p[idx])[0])
    ok_reps = [x for x in reps if np.isfinite(x)]                # separated replicates dropped (row 118)
    ci = (tuple(float(x) for x in np.percentile(ok_reps, [2.5, 97.5])) if len(ok_reps) >= 0.9 * boot
          else (float("nan"), float("nan")))
    per_set = defaultdict(list)
    for i, r in enumerate(recs):
        per_set[r["set"]].append(i)
    err = {s: float(abs(p[ix].mean() - y[ix].mean())) for s, ix in per_set.items()}
    ok = sum(e <= eps_set for e in err.values())
    need = min_sets_ok
    res = dict(auc=_auc(y, p), slope=slope, intercept=icpt, intercept_ci=ci, brier=float(((p - y) ** 2).mean()),
               brier_baseline=float(((pb - y) ** 2).mean()), per_set_error=err, sets_ok=ok,
               intercept_dropped=boot - len(ok_reps))
    res["passes"] = (res["auc"] >= H7_RULE["auc"] and H7_RULE["slope"][0] <= slope <= H7_RULE["slope"][1]
                     and ci[0] <= 0 <= ci[1]
                     and res["brier"] < res["brier_baseline"] and ok >= need)
    return res

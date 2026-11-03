"""The registered H7 scoring (Stage 2; guardrail row 125, inventory 5d).

`predictor.score` defaulted to every set passing and 1,000 bootstrap replicates; the form registers "at least 5 of 6
sets" and Stage 1 registers B = 2,000. The scoring functions now have no default for the criteria that are registered
(eps_set, boot, min_sets_ok), and the registered entry point (`predictor.score_model`, `score_registered`) takes them
from one place (`predictor.H7_RULE`, `predictor.SCORED`)."""
import inspect
import math

import numpy as np
import pytest

from reach.analysis.s2 import gp2010, gpf2013, predictor as P


def _recs(sets, shift=None, seed=3, budgets=(1, 4)):
    rng = np.random.default_rng(seed)
    out = []
    for g in sets:
        for K in range(3, 25, 2):
            for t in range(4):
                for b in budgets:
                    eta = 5 - 0.5 * K + 0.5 * math.log2(b) + (shift or {}).get(g, 0.0)
                    for s in range(3):
                        out.append(dict(set=g, target_id=f"{g}_{K}_{t}", budget=b, units=dict(nodes=float(K)),
                                        y=int(rng.random() < 1 / (1 + math.exp(-eta)))))
    return out


MODEL = dict(unit="nodes", budget_term=True, feature=None, params=[5.0, -0.5, 0.5])
BASE = dict(unit="nodes", budget_term=True, params=[4.0, -0.4, 0.3])
SIX = [f"S{i}" for i in range(6)]


def test_scoring_functions_have_no_default_for_registered_criteria():
    for fn in (P.score, P.score_p, gp2010.score, gpf2013.score, gp2010.score_beside):
        sig = inspect.signature(fn)
        for name in ("eps_set", "boot", "min_sets_ok"):
            assert sig.parameters[name].default is inspect.Parameter.empty, (fn.__name__, name)


def test_the_registered_values_are_those_filed():
    from reach.analysis.s2 import cgp_confirm, confirm
    assert P.H7_RULE["boot"] == confirm.REGISTERED["boot"] == 2000
    assert P.H7_RULE["eps_set"] == 0.10 and P.H7_RULE["auc"] == 0.80 and tuple(P.H7_RULE["slope"]) == (0.7, 1.3)
    assert P.SCORED["H7"] == dict(n_sets=6, min_sets_ok=5, budgets=(1, 2, 4, 8, 16))
    assert P.SCORED["H10b"] == P.SCORED["H7"]
    assert (cgp_confirm.EPS_SET, cgp_confirm.MIN_SETS_OK) == (P.H7_RULE["eps_set"], P.SCORED["H10b"]["min_sets_ok"])


def test_score_model_applies_five_of_six_sets_and_the_registered_bootstrap(monkeypatch):
    seen = []
    real = P.score_p

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **dict(k, boot=40))
    monkeypatch.setattr(P, "score_p", spy)
    one = P.score_model("H7", MODEL, _recs(SIX, shift={"S0": 3.0}), BASE)
    assert seen[-1]["boot"] == 2000 and seen[-1]["eps_set"] == 0.10 and seen[-1]["min_sets_ok"] == 5
    assert one["sets_ok"] == 5 and one["per_set_error"]["S0"] > 0.10 and one["test"] == "H7"
    two = P.score_model("H7", MODEL, _recs(SIX, shift={"S0": 3.0, "S1": 3.0}), BASE)
    assert two["sets_ok"] == 4 and not two["passes"]
    five = P.score_model("H7", MODEL, _recs(SIX[:5]), BASE)               # a missing set counts as failing
    assert five["n_sets_registered"] == 6 and five["sets_missing"] == 1 and five["sets_ok"] == 5


def test_score_model_refuses_records_outside_the_registered_scope():
    with pytest.raises(ValueError, match="budget"):
        P.score_model("H7", MODEL, _recs(SIX, budgets=(0.25, 1)), BASE)
    with pytest.raises(ValueError, match="sets"):
        P.score_model("H7", MODEL, _recs(SIX + ["S6"]), BASE)
    with pytest.raises(KeyError):
        P.score_model("H8", MODEL, _recs(SIX), BASE)


def test_score_beside_registered_uses_the_registered_values(monkeypatch):
    seen = []
    monkeypatch.setattr(gp2010, "score_beside", lambda *a, **k: seen.append((a, k)) or "ok")
    assert gp2010.score_beside_registered("f", "c1", "c2", "recs", "vec", "base") == "ok"
    (a, k), = seen
    assert a[6] == 0.10 and k["boot"] == 2000 and k["min_sets_ok"] == 5 and k["seed"] == P.H7_RULE["seed"]


# ------------------------------------------------------------------ D090: baselines with the budget term; benchmarks
def test_the_brier_baseline_has_the_budget_term_when_records_span_budgets():
    """D090 ("Nodes + budget"): the Brier baseline gets the same log2 budget term as the model. The choice is explicit
    (no default); with one budget the term is a constant and is refused."""
    assert inspect.signature(P.fit_baseline).parameters["budget_term"].default is inspect.Parameter.empty
    recs = _recs(SIX, seed=11, budgets=(1, 2, 4, 8, 16))
    b = P.fit_baseline(recs, budget_term=True)
    assert b["unit"] == "nodes" and b["budget_term"] is True and b["feature"] is None and len(b["params"]) == 3
    assert b["params"][1] == pytest.approx(-0.5, abs=0.08) and b["params"][2] == pytest.approx(0.5, abs=0.12)
    plain = P.fit_baseline(recs, budget_term=False)
    assert len(plain["params"]) == 2 and plain["budget_term"] is False
    assert P._logloss(b["params"], recs, "nodes", True) < P._logloss(plain["params"], recs, "nodes", False)
    with pytest.raises(ValueError, match="one budget"):
        P.fit_baseline(_recs(SIX, seed=12, budgets=(16,)), budget_term=True)


def test_benchmarks_are_described_pooled_without_pass_or_fail(monkeypatch):
    """D090 ("Descriptive, pooled"): per domain, AUC, calibration, Brier against the baseline and per-cell error."""
    seen = []
    real = P.score_p
    monkeypatch.setattr(P, "score_p", lambda *a, **k: seen.append(k) or real(*a, **dict(k, boot=40)))
    recs = [r for r in _recs(SIX, seed=13) if r["target_id"].endswith("_0")]   # one target per size: benchmark-like
    d = P.describe_model(MODEL, recs, BASE)
    assert seen[-1]["boot"] == 2000 and "passes" not in d and "sets_ok" not in d and d["descriptive"] is True
    cells = {(r["set"], r["target_id"]) for r in recs}
    assert len(d["per_cell_error"]) == len(cells)
    k = sorted(cells)[0]
    ix = [i for i, r in enumerate(recs) if (r["set"], r["target_id"]) == k]
    p = P.predict(MODEL, recs)
    assert d["per_cell_error"]["|".join(k)] == pytest.approx(abs(p[ix].mean() - np.mean([recs[i]["y"] for i in ix])))


# ------------------------------------------------------------------ guardrail row 140: degenerate calibration fits
def test_calibration_fit_treats_one_outcome_or_one_prediction_as_degenerate():
    """A bootstrap replicate with one outcome (every run exact, or none) passed the separation guard and went to the
    GLM, which gave a large finite intercept kept as a valid replicate; with one predicted value as well,
    `sm.add_constant` dropped its constant column and the fit crashed (IndexError). The registered rule (REQ-ANA-02,
    Stage 1 Missing data): a degenerate fit is NaN, a degenerate replicate is dropped and counted. Found by the Stage 2
    driver on synthetic Operon benchmark cells (D095)."""
    p = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.3])
    for y in (np.ones(6), np.zeros(6)):
        assert all(math.isnan(v) for v in P._calibration(y, p))                    # one outcome
        assert all(math.isnan(v) for v in P._calibration(y, np.full(6, 0.3)))      # one outcome, one prediction
    assert all(math.isnan(v) for v in P._calibration(np.array([1.0, 0, 1, 0]), np.full(4, 0.3)))   # one prediction
    icpt, slope = P._calibration(np.array([1.0, 0, 1, 0, 1, 0]), p)               # control: an ordinary fit
    assert math.isfinite(icpt) and math.isfinite(slope)
    # two benchmark cells with the same prediction, every run exact in one: a replicate drawing that cell alone is
    # dropped and counted (old code: IndexError)
    recs = [dict(set="S", target_id=t, budget=1, units=dict(nodes=5.0), y=y)
            for t, ys in (("a", (1, 1, 1)), ("b", (0, 1, 0))) for y in ys]
    res = P.score_p(recs, np.full(6, 0.5), np.full(6, 0.4), 0.10, boot=200, min_sets_ok=0)
    assert res["intercept_dropped"] == 200 and math.isnan(res["intercept_ci"][0]) and not res["passes"]

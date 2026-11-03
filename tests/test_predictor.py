"""H7 frozen predictor (T16; written before the code)."""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import predictor as P


def _sim(sets, shift=None, seed=1):
    """logit P = 5 - 0.5*bits + 0.5*log2(budget); bits = per_node * K; optional per-set intercept shift."""
    rng = np.random.default_rng(seed)
    recs = []
    for s, lam in sets.items():
        for K in range(3, 30, 2):
            for t in range(6):
                for b in (1, 4):
                    eta = 5 - 0.5 * lam * K + 0.5 * math.log2(b) + (shift or {}).get(s, 0.0)
                    for r in range(4):
                        recs.append(dict(set=s, target_id=f"{s}_{K}_{t}", K=K, budget=b, units=dict(nodes=float(K), bits=lam * K),
                                         y=int(rng.random() < 1 / (1 + math.exp(-eta)))))
    return recs


CAL = {"A": 0.7, "B": 1.0, "C": 1.4, "D": 1.8}
HELD = {"E": 0.85, "F": 1.6}


def test_selection_prefers_the_true_unit_and_budget_term():
    sel = P.select(_sim(CAL), units=["nodes", "bits"])
    assert sel["unit"] == "bits" and sel["budget_term"]


def test_frozen_model_passes_on_held_out_sets_that_follow_the_law(tmp_path):
    model = P.freeze(P.select(_sim(CAL), units=["nodes", "bits"]), tmp_path / "model.json")
    assert (tmp_path / "model.sha256").read_text().split()[0] == model["sha256"]
    score = P.score(model, _sim(HELD, seed=2), baseline=P.fit_baseline(_sim(CAL), budget_term=True), eps_set=0.10, boot=100, min_sets_ok=2)
    assert score["auc"] >= 0.8 and 0.7 <= score["slope"] <= 1.3 and score["intercept_ci"][0] <= 0 <= score["intercept_ci"][1]
    assert score["brier"] < score["brier_baseline"] and score["passes"]


def test_a_shifted_held_out_set_fails_the_per_set_criterion(tmp_path):
    model = P.freeze(P.select(_sim(CAL), units=["nodes", "bits"]), tmp_path / "model.json")
    score = P.score(model, _sim(HELD, shift={"F": -3.0}, seed=2), baseline=P.fit_baseline(_sim(CAL), budget_term=True), eps_set=0.10, boot=100, min_sets_ok=2)
    assert score["per_set_error"]["F"] > 0.10 and not score["passes"]


def _with_features(recs, z, noise):
    for r in recs:
        r["features"] = dict(z=z[r["set"]], noise=noise[r["set"]])
    return recs


def test_a_set_level_property_is_selected_when_it_explains_set_baselines():
    """Design 9.7 (risk fix 2026-10-25): sets differ in baseline by 1.5 * z; the model with feature z wins LOSO and
    predicts a new set's baseline; an unrelated feature is not chosen (negative control)."""
    sets = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0}
    z = {"A": -1.0, "B": -0.5, "C": 0.0, "D": 0.4, "E": 0.9, "F": 1.3}
    noise = {"A": 0.3, "B": -0.8, "C": 0.5, "D": -0.2, "E": 0.9, "F": -0.6}
    recs = _with_features(_sim(sets, shift={s: 1.5 * v for s, v in z.items()}), z, noise)
    sel = P.select(recs, ["bits"], budget_terms=(True,), features=("z", "noise"))
    assert sel["feature"] == "z" and sel["params"][-1] == pytest.approx(1.5, abs=0.4)
    plain = P.select(recs, ["bits"], budget_terms=(True,))
    assert sel["loso_logloss"] < plain["loso_logloss"]
    new = _with_features(_sim({"G": 1.0}, shift={"G": 1.5 * 1.8}, seed=9), {"G": 1.8}, {"G": 0.0})
    y = np.mean([r["y"] for r in new])
    assert abs(np.mean(P.predict(sel, new)) - y) < abs(np.mean(P.predict(plain, new)) - y)


def test_no_feature_is_chosen_when_sets_share_one_baseline():
    recs = _with_features(_sim({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}),
                          {"A": -1.0, "B": 0.0, "C": 0.5, "D": 1.0}, {"A": 0.3, "B": -0.8, "C": 0.5, "D": -0.2})
    assert P.select(recs, ["bits"], budget_terms=(True,), features=("z", "noise"))["feature"] is None


def test_freeze_never_replaces_a_frozen_model_with_other_content(tmp_path):
    """Guardrail row 123: predictor.freeze (used by freeze.freeze_model) silently replaced a frozen H7 model; the
    comparators' and the transfer model's freezes refuse (inventory 5c)."""
    path = tmp_path / "model.json"
    m = dict(unit="bits", budget_term=True, feature=None, params=[5.0, -0.5, 0.5])
    first = P.freeze(m, path)
    body, sha = path.read_bytes(), path.with_suffix(".sha256").read_bytes()
    assert P.freeze(dict(m), path)["sha256"] == first["sha256"]             # the same content: no error, no change
    with pytest.raises(RuntimeError, match="never replaced"):
        P.freeze(dict(m, params=[5.0, -0.5, 0.5000001]), path)
    assert path.read_bytes() == body and path.with_suffix(".sha256").read_bytes() == sha

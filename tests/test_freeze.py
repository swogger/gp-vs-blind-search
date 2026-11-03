"""Step 3c: freeze the predictor, place held-out levels from the frozen model only, write the prediction file."""
import json
import math

import pytest

from reach.analysis.s2 import freeze as F
from reach.analysis.s2 import g2_levels as L


MODEL = dict(unit="bits_sem", budget_term=True, params=[6.0, -0.5, 0.4],   # logit P = 6 - 0.5 s + 0.4 log2 B
             rule6=False, unit_source="H2-preferred unit (D079 a; D082)")


def test_landmarks_in_nodes_match_the_model_through_the_units_map():
    lam = 1.6                                                                 # bits per node: s = 1.6 K
    units = lambda K: dict(nodes=float(K), bits_sem=lam * K)                 # noqa: E731
    lm = F.landmarks(MODEL, units, kmax=60)
    s50 = lambda B: (6.0 + 0.4 * math.log2(B)) / 0.5                         # noqa: E731
    assert lm["k50_quarter"] == pytest.approx(s50(0.25) / lam, abs=0.05)
    assert lm["k50_16"] == pytest.approx(s50(16) / lam, abs=0.05)
    assert lm["k95"] == pytest.approx((6.0 - math.log(19)) / 0.5 / lam, abs=0.05)
    assert lm["d"] == pytest.approx(math.log(3) / 0.5 / lam, abs=0.05)


def test_heldout_levels_follow_the_g2_rule_without_heldout_data():
    units = lambda K: dict(nodes=float(K), bits_sem=1.6 * K)                 # noqa: E731
    lv = F.heldout_levels(MODEL, units, available=list(range(3, 60)))
    lm = F.landmarks(MODEL, units, kmax=60)
    ref = L.choose_k(list(range(3, 60)), lm["k95"], lm["k05"], lm["d"], lm["k50_quarter"], lm["k50_16"])
    assert (lv["main"], lv["budget"]) == (ref["main"], ref["budget"])


def test_frozen_model_and_predictions_are_checksummed_and_reproducible(tmp_path):
    m = F.freeze_model(dict(MODEL, domain="bool"), tmp_path / "model_bool.json")
    assert (tmp_path / "model_bool.sha256").read_text().split()[0] == m["sha256"]
    targets = [dict(set="S", target_id=f"t{K}", units=dict(nodes=float(K), bits_sem=1.6 * K)) for K in (5, 9, 13)]
    p = F.write_predictions(m, targets, (1, 16), tmp_path / "pred_bool.json")
    rows = json.loads(p.read_text())["predictions"]
    assert len(rows) == 6 and rows[0]["model_sha256"] == m["sha256"]
    ps = [r["p"] for r in sorted((r for r in rows if r["budget"] == 1), key=lambda r: int(r["target_id"][1:]))]
    assert ps == sorted(ps, reverse=True)                                     # larger targets, lower predictions
    assert json.loads((tmp_path / "pred_bool.json").read_text()) == json.loads(p.read_text())


def test_landmarks_handle_size_units_that_step(monkeypatch):
    """Review 2, finding 3: in binary-only sets even sizes add no functions, so bits_sem is flat between odd sizes;
    the landmarks must still be found (interpolating over the sizes where the unit rises)."""
    step = lambda K: dict(nodes=float(K), bits_sem=1.6 * (K if K % 2 else K - 1))   # noqa: E731 flat at even K
    lm = F.landmarks(MODEL, step, kmax=60)
    lin = F.landmarks(MODEL, lambda K: dict(nodes=float(K), bits_sem=1.6 * K), kmax=60)
    assert abs(lm["k50_quarter"] - lin["k50_quarter"]) < 1.0 and lm["d"] > 0


# ------------------------------------------------------------------ guardrail row 122: the set-level feature term
MODEL_F = dict(unit="bits_sem", budget_term=True, feature="z", params=[6.0, -0.5, 0.4, -3.0],
               rule6=True, unit_source="no preferred unit, rule 6: bits_sem; D082")


def _k_where(model, units, p, budget, feats, kmax=60):
    """Independent oracle: the K (fine grid) where predictor.predict, the code that scores the model, gives P = p."""
    import numpy as np
    from reach.analysis.s2 import predictor
    ks = np.linspace(1, kmax, 59001)
    recs = [dict(units=units(k), budget=budget, features=feats) for k in ks]
    return float(ks[np.argmin(np.abs(predictor.predict(model, recs) - p))])


def test_landmarks_use_the_set_level_feature_term_of_the_frozen_model():
    """Row 122: `_s_at` ignored params[3] * z, so a Boolean held-out set's levels came from another model than the one
    `predictor.predict` scores (inventory 5a). Every landmark must sit where predict() gives its probability."""
    units = lambda K: dict(nodes=float(K), bits_sem=0.8 * K)                 # noqa: E731
    for z in (0.2, 0.6):
        feats = dict(z=z)
        lm = F.landmarks(MODEL_F, units, kmax=60, features=feats)
        assert lm["k50_16"] == pytest.approx(_k_where(MODEL_F, units, 0.5, 16, feats), abs=0.01)
        assert lm["k50_quarter"] == pytest.approx(_k_where(MODEL_F, units, 0.5, 0.25, feats), abs=0.01)
        assert lm["k95"] == pytest.approx(_k_where(MODEL_F, units, 0.95, 1, feats), abs=0.01)
        assert lm["k05"] == pytest.approx(_k_where(MODEL_F, units, 0.05, 1, feats), abs=0.01)
    a, b = (F.heldout_levels(MODEL_F, units, list(range(3, 60)), kmax=60, features=dict(z=z)) for z in (0.2, 0.6))
    assert a["rule"]["feature"] == "z" and a["rule"]["feature_value"] == 0.2
    assert a["rule"]["k50_16"] - b["rule"]["k50_16"] == pytest.approx(3.0 * 0.4 / 0.5 / 0.8, abs=1e-9)


def test_a_model_with_a_feature_needs_the_sets_value():
    units = lambda K: dict(nodes=float(K), bits_sem=0.8 * K)                 # noqa: E731
    for bad in (None, {}, dict(other=1.0), dict(z=float("nan"))):
        with pytest.raises(ValueError, match="feature"):
            F.landmarks(MODEL_F, units, kmax=60, features=bad)
    F.landmarks(MODEL, units, kmax=60)                                        # control: no feature, none needed

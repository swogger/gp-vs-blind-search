"""The node-count baseline's AUC (D117; REQ-ANA-104): the same statistic as the registered scoring, on the same records."""
import json

import numpy as np
import pytest

from reach.analysis.s2 import baseline_auc, predictor
from reach.paths import ROOT


def test_auc_of_a_model_is_the_registered_auc_of_its_predictions():
    recs = [dict(y=1, K=3, budget=1, units={"nodes": 3.0, "bits_sem": 3.0}, set="A", target_id=f"t{i}") for i in range(4)] + \
           [dict(y=0, K=9, budget=1, units={"nodes": 9.0, "bits_sem": 9.0}, set="A", target_id=f"u{i}") for i in range(4)]
    model = dict(unit="nodes", budget_term=False, feature=None, params=[2.0, -0.5])      # falling in nodes: perfect ranking
    assert baseline_auc.auc_of(model, recs) == pytest.approx(1.0)
    y = np.array([r["y"] for r in recs], float)
    assert baseline_auc.auc_of(model, recs) == pytest.approx(predictor._auc(y, predictor.predict(model, recs)))
    flat = dict(unit="nodes", budget_term=False, feature=None, params=[0.0, 0.0])           # one predicted value: chance
    assert baseline_auc.auc_of(flat, recs) == pytest.approx(0.5)


@pytest.mark.study_machine
def test_recorded_baseline_auc_matches_stage2_and_the_frozen_model():
    out = json.load(open(ROOT / "data/analysis/baseline_auc.json"))
    s2 = json.load(open(ROOT / "data/analysis/stage2.json"))["tests"]["H7"]
    for d in ("real", "bool"):
        assert 0.5 < out[d]["auc"] < 1.0
        assert out[d]["frozen_auc"] == pytest.approx(s2[d]["auc"], abs=1e-9)          # the same records and model
        assert out[d]["n"] == s2[d]["n"] if "n" in s2[d] else out[d]["n"] > 0

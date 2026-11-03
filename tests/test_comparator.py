"""H7 comparator in the style of Graff, Poli and Flores (2013): problem-difficulty indicators of the target function
alone (finite differences; Boolean sensitivity), no operator-set information."""
import math

import numpy as np
import pytest

from reach import grammar as G
from reach.analysis.s2 import comparator as C


def test_boolean_features_have_known_values():
    g = G.get("BOOL-RICH-n5")
    par = C.features(g, "xor(xor(xor(xor(x1, x2), x3), x4), x5)")
    assert par["avg_sensitivity"] == 5.0 and par["degree"] == 1 and par["imbalance"] == 0.0
    dic = C.features(g, "x1")
    assert dic["avg_sensitivity"] == 1.0 and dic["essential_inputs"] == 1


def test_real_features_rise_with_wiggliness():
    g = G.get("R-TRIG")
    flat, wiggly = C.features(g, "+(x, 1)"), C.features(g, "sin(*(x, *(x, *(x, x))))")
    assert flat["turning_points"] == 0 and wiggly["turning_points"] > 0
    assert wiggly["mean_abs_d2"] > flat["mean_abs_d2"]


def test_fit_recovers_a_feature_effect_and_ignores_the_set():
    """Known answer: recovery falls with feature f, whatever the set; the comparator predicts it without set terms."""
    rng = np.random.default_rng(1)
    recs = []
    for s in ("A", "B", "C"):
        for t in range(60):
            f = rng.uniform(0, 4)
            for k in range(5):
                recs.append(dict(set=s, target_id=f"{s}{t}", budget=1, pfeatures=dict(f=f, noise=rng.normal()),
                                 y=int(rng.random() < 1 / (1 + math.exp(-(2 - 1.5 * f))))))
    m = C.fit(recs, ["f", "noise"])
    p = C.predict(m, [dict(budget=1, pfeatures=dict(f=0.0, noise=0.0)), dict(budget=1, pfeatures=dict(f=4.0, noise=0.0))])
    assert p[0] > 0.8 and p[1] < 0.05


def test_loso_logloss_is_finite_and_prefers_the_informative_feature():
    rng = np.random.default_rng(2)
    recs = [dict(set=s, target_id=f"{s}{t}", budget=1, pfeatures=dict(f=f, noise=rng.normal()),
                 y=int(rng.random() < 1 / (1 + math.exp(-(2 - 1.5 * f)))))
            for s in ("A", "B", "C") for t, f in enumerate(rng.uniform(0, 4, 60)) for _ in range(3)]
    assert C.loso_logloss(recs, ["f"]) < C.loso_logloss(recs, ["noise"])

"""T16 per-level rates with the design effect, and level suggestions (written before the code). REQ-ANA-03."""
import numpy as np
import pytest

from reach.analysis.s2 import levels as L


def test_icc_zero_and_one():
    same = [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]               # all variance between targets
    mixed = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]  # all variance within targets
    assert L.icc(same) == pytest.approx(1.0, abs=1e-9)
    assert L.icc(mixed) <= 0.0


def test_wilson_with_design_effect_is_wider():
    lo1, hi1 = L.wilson(30, 60)
    lo2, hi2 = L.wilson(30, 60, deff=3.0)
    assert lo2 < lo1 < 0.5 < hi1 < hi2
    assert L.wilson(0, 10)[0] == 0.0 and L.wilson(10, 10)[1] == pytest.approx(1.0)


def test_level_rates_use_a_floor_when_the_icc_is_undefined():
    recs = [dict(K=5, target_id=f"t{t}", y=1) for t in range(5) for _ in range(3)]          # all successes
    recs += [dict(K=7, target_id=f"u{t}", y=int(t < 2)) for t in range(5) for _ in range(3)]
    rows = {r["K"]: r for r in L.level_rates(recs)}
    assert rows[5]["rate"] == 1.0 and rows[5]["deff"] == rows[7]["deff"] >= 1.0                 # floor from the set
    assert rows[7]["rate"] == pytest.approx(0.4)


def test_suggested_levels_span_the_curve():
    ks = list(range(3, 30))
    b0, b1 = 6.0, -0.5                                                 # K50 = 12; 0.95 near K = 6, 0.05 near K = 18
    lv = L.suggest_levels(ks, b0, b1, n=10)
    assert len(lv) == 10 and min(lv) <= 7 and max(lv) >= 17 and lv == sorted(lv)

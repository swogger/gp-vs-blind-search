"""The real-domain loss beside its remedies (D100; REQ-ANA-93, 94)."""
import math

import pytest

from reach.analysis.s2 import remedy


def _runs(levels, hits_by_level, n=32):
    """Synthetic 200-generation runs: per level (bits), 5 targets x n seeds; `hits_by_level[b]` seeds hit at
    generation 10, the rest never."""
    runs, size = {}, {}
    for b in levels:
        for t in range(5):
            tid = f"b{b}_t{t}"
            size[tid] = b
            runs[tid] = [dict(target_id=tid, generations=200, gens_reached=200, terminal_state="budget_exhausted",
                              gen_exact=10 if s < hits_by_level[b] else None) for s in range(n)]
    return runs, size


def test_restart_curve_flags_plug_in_and_extrapolation():
    levels = (8.0, 10.0, 12.0)
    runs, size = _runs(levels, {8.0: 20, 10.0: 4, 12.0: 1})
    c = remedy.restart_curve(runs, size, L=100, budgets=(1, 16))
    assert c[1]["R"] == 2 and not c[1]["plug_in"]
    assert c[16]["R"] == 32 and not c[16]["plug_in"]
    # 32 restarts = all 32 seeds: every target with any hit is solved, so every level is solved: K50 = +inf
    assert c[16]["k50"] == math.inf and c[16]["extrapolated"] and c[16]["lower_bound"] == 12.0
    assert c[1]["k50"] < 12.0 and not c[1]["extrapolated"] and c[1]["lower_bound"] == c[1]["k50"]
    c25 = remedy.restart_curve(runs, size, L=25, budgets=(16,))
    assert c25[16]["R"] == 128 and c25[16]["plug_in"]


def test_rate_uses_bracketed_exact_points_only():
    pts = {1: dict(k50=10.0, extrapolated=False, plug_in=False), 2: dict(k50=11.5, extrapolated=False, plug_in=False),
           4: dict(k50=13.0, extrapolated=False, plug_in=False), 8: dict(k50=20.0, extrapolated=True, plug_in=False),
           16: dict(k50=14.0, extrapolated=False, plug_in=True)}
    r = remedy.rate_inside(pts)
    assert r["rate"] == pytest.approx(1.5) and r["points"] == 3
    assert math.isnan(remedy.rate_inside({1: pts[1], 2: pts[2], 8: pts[8]})["rate"])     # fewer than 3 usable


def test_null_gaps_cap_boolean_sizes_and_use_the_budget():
    # successes up to 30 bits, failures at "40 to 50" bits (above the 32-bit space): the sizes must be capped BEFORE the
    # fit, so the separated fit's gap midpoint is (30 + 32) / 2 = 31; capping only after the fit would give 35 -> 32
    recs = [dict(set="BOOL-AON-n5", target_id=f"t{i}", budget=1, y=int(b <= 30), units={"bits_sem": b})
            for i, b in enumerate([20, 24, 28, 30, 40, 44, 48, 50])]
    g = remedy.null_gaps(recs, budget=1)["BOOL-AON-n5"]
    assert g["k50"] == pytest.approx(31.0) and not g["capped_above"]
    assert g["null"] == pytest.approx(math.log2(20_000))
    real = [dict(set="R-TRIG", target_id=f"t{i}", budget=16, y=int(i < 5), units={"bits_sem": 8.0 + i})
            for i in range(10)]
    gr = remedy.null_gaps(real, budget=16)["R-TRIG"]
    assert gr["gap"] == pytest.approx(gr["k50"] - math.log2(16 * 20_000))


def test_eval_ratio():
    assert remedy.eval_ratio(-4.0) == pytest.approx(16.0)
    assert remedy.eval_ratio(2.0) == pytest.approx(0.25)

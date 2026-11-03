"""Gate G2 levels (docs/design/targets.md: 10 levels, 5 of them budget levels bracketing K50 from 1/4 to 16xE0)."""
import math

import pytest

from reach.analysis.s2 import g2_levels as L


def _counts(ceiling, odd_only=False, small=()):
    return [dict(K=k, U=(5 if k in small else 1000) if (k % 2 or not odd_only) else 0) for k in range(1, ceiling + 1)]


def test_available_sizes_respect_counts_parity_and_ceiling():
    av = L.available_sizes(_counts(21, odd_only=True, small=(3, 5)), kmax=31)
    assert av[0] == 7 and all(k % 2 for k in av) and max(av) == 31            # constructed above 21, odd only
    av2 = L.available_sizes(_counts(15), kmax=20)
    assert av2 == list(range(3, 21))                                             # unary operators: every size


def test_levels_are_ten_distinct_and_budget_levels_bracket_the_k50_band():
    av = list(range(3, 61))
    lv = L.choose(av, b0=6.0, b1=-0.4, k50_quarter=12.0, k50_16=20.0)
    allv = lv["main"] + lv["budget"]
    assert len(set(allv)) == 10 and len(lv["budget"]) == 5 and set(allv) <= set(av)
    k95, k05 = (math.log(19) - 6.0) / -0.4, (math.log(1 / 19) - 6.0) / -0.4
    assert min(allv) <= max(k95, 3) + 2 and max(allv) >= k05 - 2                 # covers the E0 curve
    assert min(lv["budget"]) < 12.0 and max(lv["budget"]) > 20.0                 # brackets K50 at 1/4 and 16xE0


def test_flat_budget_curve_still_gives_five_distinct_budget_levels():
    av = list(range(7, 40, 2))                                                   # odd sizes, F-EML-like
    lv = L.choose(av, b0=5.0, b1=-0.45, k50_quarter=10.8, k50_16=12.6)
    assert len(set(lv["budget"])) == 5 and len(set(lv["main"] + lv["budget"])) == 10
    assert min(lv["budget"]) < 10.8 and max(lv["budget"]) > 12.6


def test_robust_sets_are_anchor_plus_lowest_and_highest_lambda():
    lam = {"F-EML": 1.28, "R-ARITH4": 1.5, "R-ELEM4": 2.0, "R-HYB": 1.7, "R-TRIG": 1.95, "F-LDE0": 1.1}
    assert L.robust_sets(lam, anchor="F-EML") == ["F-EML", "F-LDE0", "R-ELEM4"]
    with pytest.raises(ValueError):
        L.robust_sets({"F-EML": 1.0}, anchor="F-EML")


def test_narrow_curve_with_odd_sizes_still_gets_ten_levels():
    """R-HYB-like: the E0 curve spans about 4-13 nodes and only odd sizes from 7 exist; main levels go to the tail."""
    av = L.available_sizes(_counts(15, odd_only=True, small=(3, 5)), kmax=35)
    lv = L.choose(av, b0=5.0, b1=-0.6, k50_quarter=7.7, k50_16=9.0)
    assert len(lv["main"]) == 5 and len(lv["budget"]) == 5 and len(set(lv["main"] + lv["budget"])) == 10


def test_constructed_sizes_stop_at_one_composition_above_the_ceiling():
    """Guardrail row 50: a constructed target is one operator on two minimal trees, so sizes above 2 * ceiling + 1
    cannot be built (BOOL-NAND-n5 got 0 targets at 77 with ceiling 31)."""
    av = L.available_sizes(_counts(31, odd_only=True), kmax=99)
    assert max(av) == 63
    lv = L.choose(av, b0=4.0, b1=-0.15, k50_quarter=23.1, k50_16=68.5)
    assert max(lv["budget"] + lv["main"]) <= 63 and len(set(lv["budget"] + lv["main"])) == 10

"""T10 blind-search nulls (written before the code). REQ-ANA-13."""
import math

import pytest

from reach.analysis.s2 import null as N


def _levels(us, cands=None):
    cum, out = 0, []
    for k, u in enumerate(us, 1):
        cum += u
        out.append(dict(K=k, U=u, U_cum=cum, candidates=(cands or us)[k - 1]))
    return out


LV = _levels([2, 0, 4, 0, 16, 0, 80, 0, 400])


def test_primary_null_probability_is_size_ordered():
    # E = 6 covers sizes 1 and 3 completely (2 + 4), none of size 5
    assert N.p_hit(LV, 3, 6, "enumeration") == 1.0 and N.p_hit(LV, 5, 6, "enumeration") == 0.0
    assert N.p_hit(LV, 5, 6 + 8, "enumeration") == pytest.approx(0.5)        # half of the 16 at size 5


def test_primary_null_reach_in_bits_sem_is_log2_E_up_to_one_level():
    """The null's reach on the bits_sem scale is log2 E within one size level's step (brief 4.4)."""
    bits = [math.log2(st["U_cum"]) for st in LV if st["U"] > 0]
    step = max(b - a for a, b in zip(bits, bits[1:]))                    # the coarsest level step
    for E in (10, 22, 60, 102, 300):
        assert abs(N.reach_bits_sem(LV, E) - math.log2(E)) <= step


def test_duplicates_and_syntax_cost_more():
    lv = _levels([2, 0, 4, 0, 16, 0, 80], cands=[2, 0, 8, 0, 64, 0, 640])
    E = 50
    assert N.p_hit(lv, 5, E, "bottom_up") < N.p_hit(lv, 5, E, "enumeration")


def test_reach_slope_is_one_bit_per_doubling_on_geometric_counts():
    """On counts that double per node the null gains exactly one bit per budget doubling."""
    lv = _levels([2 ** k for k in range(1, 30)])
    r = [N.reach_bits_sem(lv, 2 ** e) for e in range(8, 20)]
    assert all(abs((b - a) - 1.0) < 0.01 for a, b in zip(r, r[1:]))       # interpolation between levels


def test_uniform_draw_formula():
    p = N.p_hit(LV, 5, 10, "uniform")
    assert p == pytest.approx(1 - (1 - 1 / 22) ** 10)

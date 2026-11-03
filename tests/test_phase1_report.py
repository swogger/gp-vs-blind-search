"""Phase 1 report helpers (reach.analysis.s2.phase1), validated on known cases before use (lesson 19)."""
import numpy as np

from reach import grammar as G
from reach.analysis.s2.phase1 import lam, relative_keys
from reach.enumerate import _coeffs


def test_lambda_recovers_known_growth():
    levels = [dict(K=k, U=1, U_cum=int(3 * 2 ** (1.5 * k))) for k in range(1, 12)]
    assert abs(lam(levels) - 1.5) < 0.01
    sparse = [dict(K=k, U=(1 if k % 2 else 0), U_cum=int(2 ** (k // 2 + 1))) for k in range(1, 20)]   # NAND-like
    assert abs(lam(sparse) - 0.5) < 0.05


def test_relative_keys_merge_noise_but_not_functions():
    c = _coeffs(3 * G.N_POINTS)
    x = G.grid_sample()
    big = np.exp(np.exp(x))[None, :]                                   # values up to about 5e8
    noisy = big * (1 + 4e-16)                                          # float noise
    assert relative_keys(big, c)[0] == relative_keys(noisy, c)[0]      # merged: noise
    assert relative_keys(big, c)[0] != relative_keys(big * 1.000001, c)[0]   # distinct functions stay distinct
    five, seven = np.full((1, G.N_POINTS), 5 + 0j), np.full((1, G.N_POINTS), 7 + 0j)
    assert relative_keys(five, c)[0] != relative_keys(seven, c)[0]     # magnitude is part of the key


def test_noise_measure_separates_noise_from_distinct_functions():
    """max_rel_diff <= NOISE_REL: float noise (1e-15 relative) counts, a 1e-12 relative difference does not."""
    from reach.analysis.s2.phase1 import NOISE_REL, max_rel_diff
    x = G.grid_sample()
    a = np.exp(np.exp(x))
    assert max_rel_diff(a, a * (1 + 1e-15)) <= NOISE_REL
    assert max_rel_diff(a, a * (1 + 1e-12)) > NOISE_REL
    assert max_rel_diff(a, np.where(x.real > 2.5, np.inf, a)) == np.inf          # finiteness must agree


def test_top_excluded_skips_empty_levels():
    from reach.analysis.s2.phase1 import top_excluded
    levels = [dict(K=31, U=10, U_depth_ok=9), dict(K=32, U=0, U_depth_ok=0)]       # NAND/NOR: even level empty
    assert top_excluded(levels) == 0.1

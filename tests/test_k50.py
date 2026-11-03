"""Positive and negative controls for the K50 estimator."""
import numpy as np

from reach.analysis.k50 import k50


def _simulate(k50_true, slope, seed, n_targets=30, seeds=5, levels=range(5, 22, 2), sd_target=0.8):
    rng = np.random.default_rng(seed)
    size, y, cl = [], [], []
    for K in levels:
        for t in range(n_targets):
            u = rng.normal(0, sd_target)                      # target-level difficulty (clustering)
            p = 1 / (1 + np.exp(-(slope * (K - k50_true) + u)))
            for _ in range(seeds):
                size.append(K); y.append(rng.random() < p); cl.append(f"{K}_{t}")
    return np.array(size), np.array(y, float), cl


def test_recovers_known_k50_and_ci_covers():
    hits = 0
    for s in range(10):
        est, lo, hi, b1 = k50(*_simulate(11.0, -0.45, s), boot=200)
        assert abs(est - 11.0) < 1.0 and b1 < 0
        hits += lo <= 11.0 <= hi
    assert hits >= 8


def test_degenerate_data_gives_nan():
    est, lo, hi, b1 = k50([5, 7, 9], [1, 1, 1], ["a", "b", "c"], boot=10)
    assert np.isnan(est)


def test_single_size_level_gives_nan():
    assert np.isnan(k50([9, 9, 9, 9], [1, 0, 1, 0], ["a", "a", "b", "b"], boot=10)[0])

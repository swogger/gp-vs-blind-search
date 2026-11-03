"""H3 estimator (tool T16, `within`; REQ-ANA-04): function fixed effects on the shared-function pool, with controls."""
import math

import numpy as np

from reach.analysis.s2 import within

SETS = [f"S{i}" for i in range(6)]


def _sim(beta_within, beta_between, n_fun=40, seeds=30, seed=1):
    """Each function f has a base size m_f; set s builds it in m_f + d_fs nodes. logit P = a_f - beta_within * d_fs,
    with a_f = 2 - beta_between * (m_f - 8): size acts within functions (beta_within) or only between them."""
    rng = np.random.default_rng(seed)
    recs = []
    for f in range(n_fun):
        m = rng.integers(4, 13)
        a = 2.0 - beta_between * (m - 8) + rng.normal(0, 0.5)
        for s in rng.choice(SETS, size=rng.integers(3, 6), replace=False):
            d = int(rng.integers(0, 7))
            p = 1 / (1 + math.exp(-(a - beta_within * d)))
            for k in range(seeds):
                recs.append(dict(set=str(s), target_id=f"{s}_pool{f:03d}", seed=k, K=int(m + d), units={}, budget=1,
                                 y=int(rng.random() < p)))
    return recs


def test_pool_id_comes_from_the_target_id():
    assert within.pool_id("BOOL-NAND-n5_pool007") == "pool007"


def test_positive_control_slope_is_negative_and_share_is_high():
    recs = _sim(beta_within=0.5, beta_between=0.0)
    fe = within.fixed_effects(recs)
    assert fe["hi"] < 0 and abs(fe["beta"] - 0.5) < 0.15
    rk = within.rank_share(recs)
    assert rk["share_negative"] >= 0.7 and rk["functions"] == rk["with_spread"] + rk["no_spread"]


def test_negative_control_between_function_size_is_not_a_within_effect():
    recs = _sim(beta_within=0.0, beta_between=0.6)
    fe = within.fixed_effects(recs)
    assert fe["lo"] < 0 < fe["hi"]
    pooled = within.pooled_slope(recs)                       # without fixed effects the confound shows as a slope
    assert pooled < -0.2


def test_functions_without_variation_are_dropped_and_counted():
    recs = _sim(beta_within=0.5, beta_between=0.0, n_fun=10)
    recs += [dict(set=s, target_id=f"{s}_pool999", seed=k, K=5 + i, units={}, budget=1, y=1)
             for i, s in enumerate(SETS[:3]) for k in range(5)]
    fe = within.fixed_effects(recs)
    assert fe["dropped_constant"] == 1 and fe["functions"] == 10


def test_undefined_rank_correlations_do_not_count_as_negative():
    recs = [dict(set=s, target_id=f"{s}_pool000", seed=0, K=3 + i, units={}, budget=1, y=0)
            for i, s in enumerate(SETS[:4])]
    rk = within.rank_share(recs)
    assert rk["with_spread"] == 1 and rk["share_negative"] == 0.0 and rk["undefined"] == 1


def test_counter_instances_are_listed():
    recs = [dict(set=s, target_id=f"{s}_pool001", seed=k, K=3 + i, units={}, budget=1, y=int(k < 3 * i))
            for i, s in enumerate(SETS[:4]) for k in range(10)]
    rk = within.rank_share(recs)
    assert rk["counter_instances"] == ["pool001"]


def test_both_domains_in_one_call_are_refused():
    """Review finding 7: pool ids (pool000...) repeat across domains, so H3 must run per domain."""
    import pytest
    recs = [dict(set="BOOL-NAND-n5", target_id="BOOL-NAND-n5_pool000", K=5, units={}, y=1),
            dict(set="F-EML", target_id="F-EML_pool000", K=7, units={}, y=0)]
    with pytest.raises(ValueError):
        within.fixed_effects(recs)
    with pytest.raises(ValueError):
        within.rank_share(recs)

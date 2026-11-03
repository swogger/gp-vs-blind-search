"""H2 invariance estimator (T16), with a positive and a negative control (REQ-ANA-04)."""
import numpy as np
import pytest

from reach.analysis.s2.invariance import excess, invariance_test


def _sim(per_node, intercepts, seed=1, targets=30, seeds=5):
    """Recovery logistic in a shared 'bits' scale: logit P = a_set + 4 - 0.5 * bits, bits = per_node[set] * K."""
    rng = np.random.default_rng(seed)
    recs = []
    for s, (lam, a) in enumerate(zip(per_node, intercepts)):
        for K in range(3, 40, 2):
            bits = lam * K
            for t in range(targets // 6):
                for _ in range(seeds):
                    p = 1 / (1 + np.exp(-(a + 4 - 0.5 * bits)))
                    recs.append(dict(set=f"S{s}", target_id=f"S{s}_{K}_{t}", y=int(rng.random() < p),
                                     units=dict(nodes=float(K), bits=float(bits))))
    return recs


def test_positive_control_invariant_in_the_shared_unit_only():
    recs = _sim(per_node=[0.6, 1.0, 1.4, 2.0], intercepts=[0, 0, 0, 0])
    e_bits, e_nodes = excess(recs, "bits"), excess(recs, "nodes")
    assert e_bits < 0.01 < e_nodes                                   # same curve in bits; different in nodes
    res = invariance_test(recs, ["nodes", "bits"], delta=0.02, boot=60, seed=3)
    assert res["bits"]["invariant"] and not res["nodes"]["invariant"]


def test_negative_control_set_specific_curves_are_not_invariant():
    recs = _sim(per_node=[1.0, 1.0, 1.0, 1.0], intercepts=[-2.0, 0.0, 1.5, 3.0])
    res = invariance_test(recs, ["nodes", "bits"], delta=0.02, boot=60, seed=3)
    assert not res["bits"]["invariant"] and not res["nodes"]["invariant"]


def test_per_set_excess_names_the_shifted_set_only():
    """H2 deviating sets: one set's curve shifted; its own excess stands out, the others stay near 0."""
    from reach.analysis.s2.invariance import per_set_test
    recs = _sim(per_node=[1.0, 1.0, 1.0, 1.0], intercepts=[0, 0, 0, 2.5])
    res = per_set_test(recs, "bits", boot=40, seed=2)
    assert res["S3"]["excess"] > 3 * max(res[s]["excess"] for s in ("S0", "S1", "S2"))
    assert res["S3"]["upper95"] > res["S3"]["excess"]


def test_transfer_scores_new_sets_against_a_law_fitted_without_them():
    """H2s: sets outside `train` are scored by a fit on `train` only; a set on the same curve transfers."""
    from reach.analysis.s2.invariance import per_set_test
    recs = _sim(per_node=[1.0, 1.0, 1.0, 1.0], intercepts=[0, 0, 0, 0])
    res = per_set_test(recs, "bits", boot=30, seed=4, train=["S0", "S1", "S2"])
    assert set(res) == {"S0", "S1", "S2", "S3"} and res["S3"]["upper95"] < 0.02

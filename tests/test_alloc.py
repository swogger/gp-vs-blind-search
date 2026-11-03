"""Best split (T16; REQ-ANA-06), written before the code."""
import math

import numpy as np
import pytest

from reach.analysis.s2 import alloc as A


def test_unbiased_estimator_and_plug_in():
    # 2 successes in 4 runs; 2 runs drawn without replacement: P(at least one) = 1 - C(2,2)/C(4,2) = 1 - 1/6
    p, plug = A.p_at_least_one(2, 4, 2)
    assert p == pytest.approx(5 / 6) and not plug
    p, plug = A.p_at_least_one(1, 4, 8)                                 # more runs than seeds: plug-in, labelled
    assert p == pytest.approx(1 - (3 / 4) ** 8) and plug


def _runs(kappa, seed=1, n_seeds=32, targets=10):
    rng = np.random.default_rng(seed)
    out = []
    for K in range(6, 26, 2):
        lam = math.exp(1.5 - 0.5 * K * math.log(2))
        for t in range(targets):
            for s in range(n_seeds):
                T = (-math.log(rng.random()) / lam) ** (1 / kappa)
                g = math.ceil(T) - 1
                out.append(dict(target_id=f"K{K}_{t}", K=K, gen_exact=g if g < 200 else None, gens_reached=200,
                                generations=200, seed=s))
    return out


def test_restarts_help_only_when_kappa_below_one():
    lengths = (25, 50, 100, 200)
    dec = A.best_split(_runs(0.5), budget=4, lengths=lengths)        # decelerating search
    assert dec["best_length"] < 200 and dec["k50"][dec["best_length"]] > dec["k50"][200]
    # memoryless: no systematic gain from splitting (checked: mean difference ~0 over seeds; SD 0.18 at 60 targets)
    mem = A.best_split(_runs(1.0, targets=60), budget=4, lengths=lengths)
    assert abs(mem["k50"][25] - mem["k50"][200]) < 0.6


def test_bootstrap_reselects_the_best_split():
    res = A.best_split_ci(_runs(0.5), budget=4, lengths=(25, 50, 100, 200), boot=40, seed=2)
    assert res["lo"] <= res["k50"] <= res["hi"] and len(res["chosen_lengths"]) == 40


def test_saturated_or_error_rows_do_not_crash_the_best_split():
    """Review finding 5: rows without gen_exact (error or killed) and cells with no finite K50 are handled."""
    from reach.analysis.s2 import alloc as A
    runs = [dict(target_id=f"t{K}", K=K, gen_exact=None, generations=200) for K in range(4, 12) for _ in range(4)]
    runs.append(dict(target_id="t4", K=4, terminal_state="error"))
    r = A.best_split(runs, 1)
    assert r["best_length"] is None and math.isnan(r["best_k50"])
    ci = A.best_split_ci(runs, 1, boot=20)
    assert math.isnan(ci["lo"]) and math.isnan(ci["hi"])


def test_a_length_that_solves_every_target_is_the_best_split():
    """Review 2, finding 5: when every target is solved at some length, K50 is above the range (+inf), and that
    length must win; it was dropped as NaN."""
    from reach.analysis.s2 import alloc as A
    runs = []
    for K in range(6, 16):
        for s in range(4):
            runs.append(dict(target_id=f"t{K}", K=K, generations=200, gen_exact=150 if K > 9 else 20))
    r = A.best_split(runs, 1)
    assert r["best_length"] == 200 and r["best_k50"] == float("inf")

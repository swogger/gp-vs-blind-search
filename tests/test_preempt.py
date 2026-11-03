"""Pre-emptive analyses (D099; REQ-ANA-89 to 91): tolerance reach, weaker nulls, set offsets out of sample."""
import math

import numpy as np
import pytest

from reach.analysis.s2 import preempt


def _row(**kw):
    r = dict(set="R-X", target_id="t1", seed=1, algorithm="A", config="base", K=5, K_proven=5,
             units={"bits_sem": 7.0}, pop_size=100, generations=3200, gens_reached=3200,
             terminal_state="budget_exhausted", gen_exact=None, gen_tol=None)
    r.update(kw)
    return r


def test_tolerance_outcome_reads_gen_tol_by_the_prefix_rule():
    r = _row(gen_tol=30)                                   # within tolerance at generation 30, never exact
    out = {t["budget"]: t["y"] for t in preempt.tidy_tol([r], (0.125, 0.25, 1))}
    assert out == {0.125: 0, 0.25: 1, 1: 1}                # L = 25, 50, 200 generations
    assert preempt.tidy_tol([_row(gen_tol=30, generations=100)], (16,)) == []            # beyond the run's length
    assert preempt.tidy_tol([_row(gen_tol=30, terminal_state="error")], (1,)) == []      # errors excluded
    cut = _row(terminal_state="timeout", gens_reached=40)                                # CPU stop before L
    assert [t["censored"] for t in preempt.tidy_tol([cut], (1,))] == [True]


def test_tolerance_after_exact_is_refused():
    with pytest.raises(ValueError):
        preempt.tidy_tol([_row(gen_exact=10, gen_tol=12, terminal_state="converged", gens_reached=10)], (1,))


def _levels(ratio):
    """Levels K = 1..30 with U(K) = 2^K distinct functions and `ratio` x as many candidates."""
    out, c = [], 0
    for K in range(1, 31):
        c += 2 ** K
        out.append(dict(K=K, U=2 ** K, candidates=ratio * 2 ** K, U_cum=c))
    return out


def test_null_shifts_on_known_levels():
    E = 2 ** 20
    lv = {"a": _levels(4), "b": _levels(4)}
    # primary: reach = log2 E up to one level's granularity
    assert abs(preempt.null_shift(lv, E, "enumeration")) < 1.0
    # bottom-up evaluates 4 candidates per distinct function: it reaches about 2 bits less, so GP's gap grows by ~2
    assert preempt.null_shift(lv, E, "bottom_up") == pytest.approx(
        2.0 + preempt.null_shift(lv, E, "enumeration"), abs=0.6)
    # uniform sampling with replacement: P = 1 - exp(-E / U) crosses 1/2 at U = E / ln 2 (about 0.53 bits further)
    assert preempt.null_shift(lv, E, "uniform") == pytest.approx(
        -math.log2(1 / math.log(2)) + preempt.null_shift(lv, E, "enumeration"), abs=0.6)


def _sim(alpha, beta, n_targets=300, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_targets):
        s = rng.uniform(0, 25)
        p = 1 / (1 + math.exp(-(alpha - beta * s)))
        for k in range(10):
            out.append(dict(set="H", target_id=f"t{t}", y=int(rng.random() < p), units={"bits_sem": s}))
    return out


def test_alpha_with_fixed_beta_recovers_a_known_intercept():
    a = preempt.alpha_fixed_beta(_sim(2.0, 0.3), 0.3, "bits_sem", boot=100)
    assert a["est"] == pytest.approx(2.0, abs=0.15)
    assert a["lo"] < 2.0 < a["hi"]
    # the wrong beta moves the intercept: the fit really holds beta fixed
    assert preempt.alpha_fixed_beta(_sim(2.0, 0.3), 0.2, "bits_sem", boot=0)["est"] < 1.5


def test_offset_prediction_scores_against_the_mean():
    chosen = dict(properties=["alphabet"], coef=[4.0, -0.5])
    feats = {"A": {"alphabet": 2}, "B": {"alphabet": 4}}
    exact = preempt.offset_prediction({"A": 3.0, "B": 2.0}, feats, chosen, cal_mean=2.0)
    assert exact["rmse_chosen"] == pytest.approx(0.0) and exact["rmse_mean"] == pytest.approx(math.sqrt(0.5))
    assert exact["pred"] == {"A": 3.0, "B": 2.0}
    off = preempt.offset_prediction({"A": 2.0, "B": 2.0}, feats, chosen, cal_mean=2.0)
    assert off["rmse_chosen"] > off["rmse_mean"] == 0.0


def test_alpha_with_fixed_beta_converges_from_far_away():
    """Guardrail row 153: Newton from alpha = 0 on held-out data (sizes 8 to 25 bits, beta 0.356, alpha about 4)
    overshot to where every p rounds to 1 and divided by a zero curvature."""
    rng = np.random.default_rng(5)
    recs = []
    for t in range(150):
        s = rng.uniform(8, 25)
        p = 1 / (1 + math.exp(-(4.0 - 0.356 * s)))
        recs += [dict(set="H", target_id=f"t{t}", y=int(rng.random() < p), units={"bits_sem": s}) for _ in range(10)]
    a = preempt.alpha_fixed_beta(recs, 0.356, "bits_sem", boot=50)
    assert a["est"] == pytest.approx(4.0, abs=0.3) and a["dropped"] == 0
    one = [dict(r, y=1) for r in recs[:20]]
    assert math.isnan(preempt.alpha_fixed_beta(one, 0.356, "bits_sem", boot=0)["est"])   # one outcome: no finite value


def test_gap_against_a_weaker_null_keeps_the_registered_primary_gap():
    """The primary null's row is H6c's registered gap exactly; a weaker null shifts it by the difference of the two
    nulls' computed reach, so the level granularity of the computed reach cancels."""
    lv = {"a": _levels(4), "b": _levels(4)}
    g6 = dict(gap=-4.0, lo=-4.5, hi=-3.5)
    assert preempt.gap_against(g6, lv, 2 ** 20, "enumeration") == dict(gap=-4.0, lo=-4.5, hi=-3.5, shift=0.0)
    bu = preempt.gap_against(g6, lv, 2 ** 20, "bottom_up")
    assert bu["shift"] == pytest.approx(2.0, abs=0.6) and bu["gap"] == pytest.approx(-4.0 + bu["shift"])
    assert bu["hi"] - bu["lo"] == pytest.approx(1.0)

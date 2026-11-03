"""Readings of the filed Stage 1 text (D079; REQ-ANA-30 to 33; guardrail rows 97 to 100), written before the code.
Synthetic data only; no calibration outcome is read.

(a) H2's preferred unit: hypotheses.md at the registered commit 3b8aa5e, "4. If several units are invariant, the
    preferred one has the lowest LOGO log-loss." and "6. If none is invariant, report the ranking of sets in bits_sem".
(b) H3: "an undefined correlation counts as not negative" (all functions stay in the denominator).
(c) The pooled budget law: set intercepts with a common slope, so gamma/beta is "the slope of K50 on log2 budget".
H2c: "if more than 10% [of replicates] are dropped, the CI is not estimable and the hypothesis is not supported"; a K50
whose slope is not negative is NaN.
"""
import math

import numpy as np
import pytest

from reach.analysis.s2 import confirm as C
from reach.analysis.s2 import invariance as I
from reach.analysis.s2 import law as L
from reach.analysis.s2 import within as W

# ------------------------------------------------------------------ (a) H2's preferred unit


def _inv(p, logo, excess):
    return {u: dict(excess=excess[u], upper95=excess[u] + 0.001, invariant=p[u] < 0.05, boot=2000,
                    p_below_delta=p[u], logo=logo[u]) for u in C.UNITS}


def test_preferred_unit_is_the_lowest_logo_loss_among_holm_invariant_units():
    """nodes and bits_uniform pass Holm (p 0, 0); bits_sem has p 0.03 (Holm 2 x 0.03 = 0.06: not invariant) but the
    lowest LOGO loss; nodes has the smallest excess. Rule 4 picks bits_uniform. (Planted: the smallest excess picks
    nodes; an unadjusted decision picks bits_sem.)"""
    inv = _inv(p=dict(nodes=0.0, bits_uniform=0.0, bits_syn=0.5, bits_sem=0.03),
               logo=dict(nodes=0.40, bits_uniform=0.30, bits_syn=0.20, bits_sem=0.10),
               excess=dict(nodes=0.001, bits_uniform=0.004, bits_syn=0.02, bits_sem=0.003))
    pref = I.preferred_unit(inv, alpha=0.05)
    assert pref["preferred"] == "bits_uniform" and pref["unit"] == "bits_uniform"
    assert pref["invariant"] == ["nodes", "bits_uniform"] and pref["rule"].startswith("4")
    one = I.preferred_unit(_inv(p=dict(nodes=0.5, bits_uniform=0.5, bits_syn=0.5, bits_sem=0.0),
                                logo=dict(nodes=0.1, bits_uniform=0.1, bits_syn=0.1, bits_sem=0.9),
                                excess=dict(nodes=0.0, bits_uniform=0.0, bits_syn=0.0, bits_sem=0.0)))
    assert one["preferred"] == "bits_sem" and one["invariant"] == ["bits_sem"]       # the only invariant unit
    tie = I.preferred_unit(_inv(p=dict(nodes=0.0, bits_uniform=0.0, bits_syn=0.5, bits_sem=0.5),
                                logo=dict(nodes=0.2, bits_uniform=0.2, bits_syn=0.1, bits_sem=0.1),
                                excess=dict(nodes=0.0, bits_uniform=0.0, bits_syn=0.0, bits_sem=0.0)))
    assert tie["preferred"] == "nodes"                                                # a tie: the first in order


def test_no_invariant_unit_gives_no_preferred_unit_and_bits_sem_ranking():
    """Rule 6: no unit invariant (an unadjusted p of 0.02 does not pass Holm at m = 4) -> no preferred unit; the
    per-set report is in bits_sem."""
    inv = _inv(p=dict(nodes=0.5, bits_uniform=0.2, bits_syn=0.3, bits_sem=0.02),
               logo=dict(nodes=0.1, bits_uniform=0.2, bits_syn=0.3, bits_sem=0.4),
               excess=dict(nodes=0.0, bits_uniform=0.0, bits_syn=0.0, bits_sem=0.0))
    pref = I.preferred_unit(inv, alpha=0.05)
    assert pref["preferred"] is None and pref["unit"] == "bits_sem" and pref["invariant"] == []
    assert pref["rule"].startswith("6")


def test_invariance_reports_mean_logo_logloss():
    """'LOGO log-loss' of a unit = the mean over sets of each set's leave-one-set-out log-loss; excess = LOGO - within,
    so LOGO - excess is the mean within-set loss. A shared curve in 'good' gives a lower LOGO loss than 'bad'."""
    rng = np.random.default_rng(7)
    recs = []
    for gi, shift in enumerate([0, 3, 6, -3]):
        for K in range(4, 30, 3):
            for t in range(6):
                p = 1 / (1 + math.exp(-(8 - 0.5 * (K + shift))))
                for s in range(5):
                    recs.append(dict(set=f"S{gi}", target_id=f"S{gi}_{K}_{t}", y=int(rng.random() < p),
                                     units=dict(good=float(K + shift), bad=float(K))))
    res = I.invariance_test(recs, ["good", "bad"], delta=0.01, boot=20, seed=1)
    assert res["good"]["logo"] < res["bad"]["logo"]
    # direct LOGO loss of each set from a fit on the others
    by = I._arrays(recs, "bad")
    sets = sorted(by)
    logos = []
    for s in sets:
        xo = np.array([v[0] for o in sets if o != s for t in by[o] for v in by[o][t]], float)
        yo = np.array([v[1] for o in sets if o != s for t in by[o] for v in by[o][t]], float)
        xs = np.array([v[0] for t in by[s] for v in by[s][t]], float)
        ys = np.array([v[1] for t in by[s] for v in by[s][t]], float)
        logos.append(I._ll(I._fit(xo, yo), xs, ys))
    assert res["bad"]["logo"] == pytest.approx(float(np.mean(logos)), rel=1e-12)
    assert res["bad"]["logo"] - res["bad"]["excess"] == pytest.approx(res["bad"]["within"], rel=1e-9)


def _cal_root(tmp_path):
    from test_dryrun import SETS, _rows, _write
    _write(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + 2 * i, i)])
    return tmp_path


def test_confirm_uses_the_registered_preferred_unit_rule(tmp_path, monkeypatch):
    """confirm.build takes H2's preferred unit from the rule (D079 a), and uses it for the per-set excesses; with no
    invariant unit it reports the sets' bits_sem ranking and works in bits_sem, labelled."""
    root = _cal_root(tmp_path)
    used = []
    real_per_set = I.per_set_test
    monkeypatch.setattr(I, "per_set_test", lambda recs, unit, **kw: used.append(unit) or real_per_set(recs, unit, **kw))
    inv = _inv(p=dict(nodes=0.0, bits_uniform=0.0, bits_syn=0.5, bits_sem=0.03),
               logo=dict(nodes=0.40, bits_uniform=0.30, bits_syn=0.20, bits_sem=0.10),
               excess=dict(nodes=0.001, bits_uniform=0.004, bits_syn=0.02, bits_sem=0.003))
    monkeypatch.setattr(I, "invariance_test", lambda *a, **k: inv)
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), root, boot=20, allow_incomplete=True)
    h = res["h2"]["bool"]
    assert h["preferred"] == "bits_uniform" and h["report_unit"] == "bits_uniform" and used[0] == "bits_uniform"
    assert h["invariant_holm"] == ["nodes", "bits_uniform"] and res["sensitivity_lower_bound"]["bool"]["preferred"] == \
        "bits_uniform"
    none = _inv(p=dict(nodes=0.5, bits_uniform=0.5, bits_syn=0.5, bits_sem=0.5),
                logo=dict(nodes=0.1, bits_uniform=0.2, bits_syn=0.3, bits_sem=0.4),
                excess=dict(nodes=0.0, bits_uniform=0.0, bits_syn=0.0, bits_sem=0.0))
    monkeypatch.setattr(I, "invariance_test", lambda *a, **k: none)
    used.clear()
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), root, boot=20, allow_incomplete=True)
    h = res["h2"]["bool"]
    assert h["preferred"] is None and h["report_unit"] == "bits_sem" and used[0] == "bits_sem"
    rk = h["ranking_bits_sem"]
    assert [x["k50"] for x in rk] == sorted([x["k50"] for x in rk], reverse=True) and len(rk) == 3
    assert rk[0]["set"] == "BOOL-AON-n5"                       # the easiest set (largest intercept) reaches furthest
    assert "rule 6" in C.render(res, "t", "s")


# ------------------------------------------------------------------ (b) H3's rank share


def _pool(f, s, K, rate, n=10):
    return [dict(set=s, target_id=f"{s}_{f}", K=K, units={}, y=int(i < rate * n)) for i in range(n)]


def test_rank_share_keeps_undefined_tau_in_the_denominator():
    """One negative, one without size spread (tau undefined), one positive function: the filed share is 1/3; the
    spread-only share (exploratory) is 1/2. (Planted: the previous denominator gives 1/2 as the registered share.)"""
    recs = (_pool("pool1", "SA", 5, 0.9) + _pool("pool1", "SB", 9, 0.2) + _pool("pool1", "SC", 13, 0.1)
            + _pool("pool2", "SA", 7, 0.5) + _pool("pool2", "SB", 7, 0.4) + _pool("pool2", "SC", 7, 0.6)
            + _pool("pool3", "SA", 5, 0.1) + _pool("pool3", "SB", 9, 0.5) + _pool("pool3", "SC", 13, 0.9))
    rk = W.rank_share(recs)
    assert rk["functions"] == 3 and rk["negative"] == 1 and rk["no_spread"] == 1
    assert rk["share_negative"] == pytest.approx(1 / 3)
    assert rk["share_negative_spread_only"] == pytest.approx(1 / 2)
    assert rk["counter_instances"] == ["pool3"]


# ------------------------------------------------------------------ (c) the pooled budget law


def _two_sets(seed=1):
    """Two sets of different difficulty, each with its levels around its own K50; true rate gamma/beta = 1.0."""
    rng = np.random.default_rng(seed)
    rows = []
    for gi, (a, levels) in enumerate([(5.0, [6, 8, 10, 12, 14]), (15.0, [26, 28, 30, 32, 34])]):
        for K in levels:
            for t in range(30):
                for b in (1, 2, 4, 8, 16):
                    p = 1 / (1 + math.exp(-(a - 0.5 * K + 0.5 * math.log2(b))))
                    for s in range(5):
                        rows.append(dict(set=f"S{gi}", target_id=f"S{gi}_{K}_{t}", K=K, units={"bits_sem": float(K)},
                                         budget=b, y=int(rng.random() < p)))
    return rows


def test_pooled_law_with_set_intercepts_recovers_the_k50_slope():
    """Known answer 1.0 bits per doubling: set intercepts recover it; one intercept does not (D079 c; audit: 11.8)."""
    rows = _two_sets()
    two = L.fit_law(rows, "bits_sem", set_specific=True)
    one = L.fit_law(rows, "bits_sem")
    assert two["rate"] == pytest.approx(1.0, abs=0.15) and two["intercepts"] == "set"
    assert set(two["alphas"]) == {"S0", "S1"} and two["alphas"]["S1"] > two["alphas"]["S0"] + 5
    assert abs(one["rate"] - 1.0) > 2 and one["intercepts"] == "one"
    assert two["se_beta_cluster"] > 0 and two["targets"] == 300


def test_rate_ci_and_curvature_use_set_intercepts():
    rows = _two_sets()
    rc = L.rate_ci(rows, "bits_sem", boot=60, set_specific=True)
    assert rc["lo"] < 1.0 < rc["hi"] and rc["model"] == "set intercepts, common slope"
    assert rc["rate"] == pytest.approx(L.fit_law(rows, "bits_sem", set_specific=True)["rate"])
    one = L.rate_ci(rows, "bits_sem", boot=60)
    assert not one["lo"] < 1.0 < one["hi"] and one["model"] == "one intercept"
    cv = L.curvature(rows, "bits_sem", set_specific=True)
    assert cv["lo"] < 0 < cv["hi"] and cv["model"] == "set intercepts, common slope"     # no quadratic term


def test_confirm_h4_reports_set_intercepts_and_the_one_intercept_fit(tmp_path):
    root = _cal_root(tmp_path)
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), root, boot=20, allow_incomplete=True)
    h = res["h4"]["bool"]
    for k in ("law_nodes", "law_bits_sem"):
        assert h[k]["intercepts"] == "set" and h[k + "_one_intercept"]["intercepts"] == "one"
        assert set(h[k]["alphas"]) == {"BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5"}
    assert h["rate_bits_sem"]["model"] == "set intercepts, common slope"
    assert h["rate_bits_sem_one_intercept"]["model"] == "one intercept"
    assert h["curvature_nodes"]["model"] == "set intercepts, common slope"
    assert h["curvature_nodes_one_intercept"]["model"] == "one intercept"
    assert h["rate_bits_sem"]["rate"] == pytest.approx(h["law_bits_sem"]["rate"])
    mem = {m["id"][0]: m for m in res["holm"]["H4"]["bool"]["members"]}
    assert mem["H4a"]["p"] == h["curvature_nodes"]["p_wald"]                # Holm reads the registered model
    six = {m["id"][0]: m for m in res["holm"]["H6"]["bool"]["members"]}
    assert six["H6a"]["p"] == h["rate_bits_sem"]["p_differs1"]


# ------------------------------------------------------------------ H2c


def _h2c_sample(name, ks_y, runs=4):
    """Records of one sample: per target (size, successes out of `runs`)."""
    out = []
    for i, (K, s) in enumerate(ks_y):
        out += [dict(set=name, target_id=f"{name}_{i}", K=K, units=dict(nodes=float(K)), y=int(j < s))
                for j in range(runs)]
    return out


def test_h2c_k50_with_a_rising_curve_is_nan():
    """A sample whose recovery rises with size has no K50 (slope not negative): the difference is NaN, not a number,
    and H2c does not hold. (Planted: -b0/b1 without the sign check gives a finite difference.)"""
    rising = _h2c_sample("a", [(5, 0), (5, 1), (10, 2), (10, 2), (15, 3), (15, 4)])
    falling = _h2c_sample("b", [(5, 4), (5, 3), (10, 2), (10, 2), (15, 1), (15, 0)])
    d = C.k50_difference(rising, falling, boot=50)
    assert math.isnan(d["diff"]) and not d["holds"] and not d["estimable"]
    # Control: three times the falling sample, so that no replicate is separated (with 6 targets, 15 of 50 replicates
    # were, and a separated replicate has no K50: guardrail row 118)
    f3 = [(5, 4), (5, 3), (10, 2), (10, 2), (15, 1), (15, 0)] * 3
    ok = C.k50_difference(_h2c_sample("b", f3), _h2c_sample("c", f3), boot=50)
    assert ok["diff"] == pytest.approx(0.0, abs=1e-9) and ok["estimable"] and ok["dropped"] == 0


def test_h2c_more_than_10_percent_degenerate_replicates_is_not_supported():
    """Two samples of 4 targets at sizes 5 and 15 (3 of 4 and 1 of 4 runs exact): K50 is exactly 10 whenever both sizes
    are drawn, and degenerate (one size only) in 2/16 of draws per sample, about 23% of replicates. The CI is [0, 0],
    inside the 1-node margin, but more than 10% are dropped: not estimable, not supported. (Planted: ignoring
    `dropped` makes H2c hold.)"""
    a = _h2c_sample("a", [(5, 3), (5, 3), (15, 1), (15, 1)])
    b = _h2c_sample("b", [(5, 3), (5, 3), (15, 1), (15, 1)])
    d = C.k50_difference(a, b, boot=400, seed=3)
    assert d["diff"] == pytest.approx(0.0, abs=1e-9)
    assert abs(d["lo"]) < 1e-6 and abs(d["hi"]) < 1e-6                        # inside the margin
    assert 0.15 < d["dropped"] / 400 < 0.32 and not d["estimable"] and not d["holds"]

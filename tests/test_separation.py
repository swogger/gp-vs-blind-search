"""Separation in the logistic fits (guardrail row 118, amended by D088; REQ-ANA-02).

Complete or quasi-complete separation (successes and failures do not overlap in size, except at one shared value):
the maximum-likelihood slope of a one-predictor fit is -inf (successes at the smaller sizes) or +inf. Row 118 (D086)
made such a fit "not estimable" and dropped separated replicates. D088 (author, 2026-10-27, "keep the sign"): a
one-predictor slope keeps its sign (point estimate and replicates +-inf, kept in the percentile CI and in the
CI-inversion p) and a one-predictor K50 is the midpoint of the gap between the largest size with a success and the
smallest size with a failure (quasi-complete at a shared size s: K50 = s). Truly degenerate fits (one outcome, one
size, singular) are still dropped. Multi-predictor fits, H2's log-loss fits, H8c's base curves and H10b's transfer fit
keep row 118's rule (D088 says why). Planted cases from QA check 3 (D068, D086)."""
import math

import numpy as np
import pytest

from reach.analysis import k50 as K50
from reach.analysis.s2 import alloc, cfgint, contrasts, invariance, law, within

INF = float("inf")
COMPLETE = ([3] * 10 + [4] * 10 + [6] * 10 + [7] * 10, [1] * 20 + [0] * 20)          # 3, 4 succeed; 6, 7 fail
QUASI = ([3] * 10 + [5] * 10 + [7] * 10, [1] * 10 + [1, 0] * 5 + [0] * 10)          # 3 succeed; 5 half; 7 fail
RISING = ([3] * 10 + [4] * 10 + [6] * 10 + [7] * 10, [0] * 20 + [1] * 20)            # the wrong direction


def _cell(xy, per_target=5):
    x, y = xy
    return [dict(set="S", target_id=f"t{i // per_target}", seed=i % per_target, K=k, budget=1,
                 units=dict(nodes=float(k), bits_sem=2.0 * k), y=v) for i, (k, v) in enumerate(zip(x, y))]


def test_separated_is_exact():
    assert K50.separated(*COMPLETE) and K50.separated(*QUASI)
    assert K50.separated([3, 4, 6], [0, 0, 1])                                       # rising: also separated
    assert not K50.separated([3, 4, 5, 6], [1, 0, 1, 0])                             # overlap
    assert not K50.separated([3, 5, 5, 7], [1, 0, 1, 1])                             # a failure inside the successes
    assert K50.separated([3, 5, 7], [1.0, 0.5, 0.0])                                 # rates: 0.5 is both outcomes
    assert not K50.separated([3, 5, 7], [1.0, 0.5, 0.25])
    assert not K50.separated([3, 4], [1, 1])                                         # one outcome: degenerate, not this
    assert K50.mle(*COMPLETE) is None and K50.mle(*QUASI) is None                    # row 118's fitter is unchanged
    b = K50.mle([3, 4, 5, 6], [1, 0, 1, 0])
    assert b is not None and np.allclose(b, K50.logit([3, 4, 5, 6], [1, 0, 1, 0]))
    assert K50.separation(*COMPLETE) == -1 and K50.separation(*QUASI) == -1 and K50.separation(*RISING) == 1
    assert K50.separation([3, 4, 5, 6], [1, 0, 1, 0]) == 0 and K50.separation([3, 4], [1, 1]) == 0


def test_fit1_gives_the_signed_limit_and_the_gap_midpoint():
    """D088's one-predictor fit: separated falling -> slope -inf, K50 the gap midpoint; quasi-complete at a shared size
    s -> K50 = s; rising -> +inf with the analogous midpoint; degenerate -> None; otherwise exactly `logit`."""
    f = K50.fit1(*COMPLETE)
    assert f.b1 == -INF and f.k50 == 5.0 and f.separated == -1                       # (4 + 6) / 2
    f = K50.fit1([3, 4, 7, 8] * 3, [1, 1, 0, 0] * 3)
    assert f.b1 == -INF and f.k50 == 5.5                                             # (4 + 7) / 2, not (3 + 8) / 2
    f = K50.fit1(*QUASI)
    assert f.b1 == -INF and f.k50 == 5.0                                             # the shared size
    assert K50.fit1([3, 5, 7], [1.0, 0.5, 0.0]).k50 == 5.0                           # rates
    f = K50.fit1([3, 4, 6, 9], [0, 0, 1, 1])
    assert f.b1 == INF and f.k50 == 5.0 and f.separated == 1                         # (4 + 6) / 2
    for x, y in (([3, 4], [1, 1]), ([3, 4], [0, 0]), ([5, 5, 5], [1, 0, 1]), ([], [])):
        assert K50.fit1(x, y) is None                                                # truly degenerate: still dropped
    x, y = [3, 4, 5, 6, 7], [1, 1, 0, 1, 0]
    f, b = K50.fit1(x, y), K50.logit(x, y)
    assert f.separated == 0 and f.b0 == b[0] and f.b1 == b[1] and f.k50 == -b[0] / b[1]


def test_percentile_on_the_extended_reals():
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.normal(size=rng.integers(2, 60))
        assert np.array_equal(law.percentile(x, [2.5, 97.5]), np.percentile(x, [2.5, 97.5]))   # bit for bit
    x = [-INF] * 3 + [1.0, 2.0, 3.0, 4.0]
    lo, hi = law.percentile(x, [2.5, 97.5])
    assert lo == -INF and hi == np.percentile([-9.0] * 3 + [1.0, 2.0, 3.0, 4.0], 97.5)
    assert tuple(law.percentile([-INF] * 10, [2.5, 97.5])) == (-INF, -INF)
    assert law.percentile([-INF, 1.0], [50])[0] == -INF and law.percentile([1.0, INF], [50])[0] == INF
    assert law.percentile([-INF, 1.0], [0])[0] == -INF and law.percentile([-INF, 1.0], [100])[0] == 1.0
    assert math.isnan(law.percentile([-INF, INF], [50])[0])                          # no interpolation between them


def _p_by_search(reps, null, direction, sided):
    """Independent oracle: the smallest alpha at which the percentile bound lies beyond null, by bisection (the bound
    moves monotonically towards the centre as alpha grows)."""
    def beyond(a):
        q = (100 * (1 - a / sided)) if direction == "below" else 100 * a / sided
        v = law.percentile(reps, [q])[0]
        return bool((v < null) if direction == "below" else (v > null))
    if beyond(0.0):
        return 0.0
    if not beyond(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if beyond(mid) else (mid, hi)
    return hi


def test_ci_inversion_p_with_infinities():
    assert law.p_ci_inversion([-INF] * 10) == 0.0
    assert law.p_ci_inversion([-INF] * 5 + [1.0] * 5) == pytest.approx(2 * (1 - 5 / 9))
    assert law.p_ci_inversion([INF] * 10) == 1.0
    rng = np.random.default_rng(2)
    for _ in range(25):
        n = int(rng.integers(5, 40))
        reps = list(rng.normal(-0.5, 1, n))
        for j in rng.choice(n, int(rng.integers(1, n)), replace=False):
            reps[j] = -INF if rng.random() < 0.7 else INF
        for direction, sided in (("below", 2), ("below", 1), ("above", 2)):
            p = law.p_bound(reps, 0.0, direction, sided)
            assert p == pytest.approx(_p_by_search(reps, 0.0, direction, sided), abs=1e-9)
    from reach.analysis.s2.families import p_differs
    assert p_differs([-INF] * 10, 0.0) == 0.0 and p_differs([INF] * 10, 0.0) == 0.0


@pytest.mark.parametrize("xy", [COMPLETE, QUASI], ids=["complete", "quasi"])
def test_h1_planted_separated_cell_keeps_its_sign_and_is_supported(xy):
    """QA's planted cells: 3, 4 succeed / 6, 7 fail, and 3 succeed / 5 half / 7 fail: slope -inf, estimable, the CI
    below 0, p 0, supported after Holm; replicates are -inf or (one outcome or one size) dropped."""
    s = law.slope_ci(_cell(xy), boot=200)
    assert s["slope"] == -INF and s["estimable"] and s["hi"] < 0 and s["lo"] == -INF
    assert s["separated"] == "falling" and s["separated_reps"] + s["dropped"] == 200 and s["dropped"] < 20
    assert s["p_below0"] == 0.0
    cell = dict(set="S", algorithm="A", config="base", budget=1, **s)
    h = law.holm_cells([cell], lambda g: "bool", 0.05)
    assert cell["holm_supported"] and h["bool"]["supported"] == 1


def test_h1_quasi_planted_cell_has_k50_at_the_shared_size():
    assert law._k50(_cell(QUASI), "nodes") == 5.0 and law._k50(_cell(QUASI), "bits_sem") == 10.0


def test_h1_wrong_direction_keeps_its_sign_and_is_not_supported():
    s = law.slope_ci(_cell(RISING), boot=200)
    assert s["slope"] == INF and s["estimable"] and s["lo"] > 0 and s["separated"] == "rising"
    assert s["p_below0"] == 1.0
    cell = dict(set="S", algorithm="A", config="base", budget=1, **s)
    law.holm_cells([cell], lambda g: "bool", 0.05)
    assert not cell["holm_supported"] and cell["holm_status"] == "not supported"


def test_h1_truly_degenerate_cells_stay_not_estimable():
    one_outcome = law.slope_ci(_cell(([3] * 10 + [6] * 10, [1] * 20)), boot=50)
    assert math.isnan(one_outcome["slope"]) and not one_outcome["estimable"] and one_outcome["dropped"] == 50
    assert one_outcome["saturated"] and "separated" not in one_outcome
    one_size = law.slope_ci(_cell(([5] * 20, [1, 0] * 10)), boot=50)
    assert math.isnan(one_size["slope"]) and not one_size["estimable"] and one_size["dropped"] == 50
    assert math.isnan(one_size["p_below0"]) and "separated_reps" not in one_size


def _overlap_cell(n=10, extra=3):
    """Separated except for `extra` targets at size 7 with one success in 5 seeds."""
    recs, t = [], 0
    for K, y in ((3, 1), (4, 1), (6, 0), (7, 0)):
        for _ in range(n):
            recs += [dict(set="S", target_id=f"t{t}", seed=s, K=K, budget=1, units=dict(nodes=float(K)), y=y)
                     for s in range(5)]
            t += 1
    star = []
    for _ in range(extra):
        recs += [dict(set="S", target_id=f"t{t}", seed=s, K=7, budget=1, units=dict(nodes=7.0), y=int(s == 0))
                 for s in range(5)]
        star.append(t)
        t += 1
    return recs, star, n


def _replay(n_ids, boot, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_ids, n_ids) for _ in range(boot)]


def _draw_xy(recs, draw):
    by = {}
    for r in recs:
        by.setdefault(r["target_id"], []).append(r)
    ids = list(by)
    rr = [r for j in draw for r in by[ids[j]]]
    return np.array([r["K"] for r in rr], float), np.array([r["y"] for r in rr], float)


def _kind(x, y):
    """Independent classification of one replicate: 'degenerate', 'falling', 'rising' or 'overlap'."""
    if y.min() == y.max() or x.min() == x.max():
        return "degenerate"
    s, f = x[y > 0], x[y < 1]
    return "falling" if s.max() <= f.min() else "rising" if f.max() <= s.min() else "overlap"


@pytest.mark.parametrize("extra", [1, 3])
def test_h1_separated_replicates_are_kept_as_infinities(extra):
    """Each replicate recomputed independently: one outcome or one size -> dropped; separated -> -inf (+inf), kept;
    else the IRLS slope. The CI is the percentile of the kept replicates."""
    recs, star, n = _overlap_cell(extra=extra)
    s = law.slope_ci(recs, boot=200, seed=7)
    kept, nan = [], 0
    for d in _replay(4 * n + extra, 200, 7):
        x, y = _draw_xy(recs, d)
        k = _kind(x, y)
        if k == "degenerate":
            nan += 1
        else:
            kept.append(-INF if k == "falling" else INF if k == "rising" else K50.logit(x, y)[1])
    sep = sum(np.isinf(v) for v in kept)
    assert 0 < sep and s["dropped"] == nan and s["separated_reps"] == sep
    assert s["estimable"] == (nan <= 20) and np.isfinite(s["slope"]) and s["slope"] < 0
    low = sum(v == -INF for v in kept)
    h = 0.975 * (len(kept) - 1)
    if low <= math.floor(h) and low == sep:                                     # hi's neighbours are finite
        finite = [v for v in kept if np.isfinite(v)]
        oracle = [v if np.isfinite(v) else min(finite) - 1 for v in kept]      # same order, finite stand-ins
        assert s["hi"] == np.percentile(oracle, 97.5)
    if low > math.ceil(0.025 * (len(kept) - 1)):
        assert s["lo"] == -INF


def test_k50_keeps_separated_fits_and_replicates_at_the_gap_midpoint():
    x, y = COMPLETE
    est, lo, hi, b1 = K50.k50(x, y, [f"t{i // 5}" for i in range(len(x))], boot=20)
    assert est == 5.0 and b1 == -INF and lo <= 5.0 <= hi
    est, _, _, b1 = K50.k50(*RISING, [f"t{i // 5}" for i in range(40)], boot=0)
    assert est == 5.0 and b1 == INF                                                 # the caller's rising handling
    recs, star, n = _overlap_cell(extra=3)
    size, yy, cl = [r["K"] for r in recs], [r["y"] for r in recs], [r["target_id"] for r in recs]
    est, lo, hi, b1 = K50.k50(size, yy, cl, boot=200, seed=11)
    keep = []
    for d in _replay(len(set(cl)), 200, 11):
        xx, y2 = _draw_xy(recs, d)
        k = _kind(xx, y2)
        if k == "falling":
            keep.append((xx[y2 > 0].max() + xx[y2 < 1].min()) / 2)
        elif k == "overlap":
            b = K50.logit(xx, y2)
            if b[1] < 0:
                keep.append(-b[0] / b[1])
    assert len(keep) > 150 and (lo, hi) == pytest.approx(tuple(np.percentile(keep, [2.5, 97.5])), abs=1e-9)


@pytest.mark.parametrize("xy", [COMPLETE, QUASI], ids=["complete", "quasi"])
def test_every_one_predictor_k50_uses_the_gap_midpoint(xy):
    x, y = xy
    assert law._k50(_cell(xy), "nodes") == 5.0 and law._k50(_cell(xy), "bits_sem") == 10.0
    assert contrasts._k50(x, y) == 5.0
    from reach.analysis.s2.confirm import _k50_signed
    assert _k50_signed(x, y, [f"t{i // 5}" for i in range(len(x))]) == 5.0
    assert alloc._k50([3, 4, 6, 7], [1.0, 1.0, 0.0, 0.0]) == 5.0                   # the best split's rates
    assert alloc._k50([3, 5, 7], [1.0, 0.5, 0.0]) == 5.0
    assert alloc._k50([3, 4, 7, 8], [1.0, 1.0, 0.0, 0.0]) == 5.5
    assert alloc._k50([3, 4, 7, 8], [1.0, 0.75, 0.0, 0.0]) == 4.0                  # quasi at the shared size 4
    assert np.isfinite(alloc._k50([3, 5, 7, 9], [1.0, 0.25, 0.75, 0.0]))
    # H8c's base curve keeps row 118 (D088: its log-loss has no limit that does not depend on the clip)
    assert cfgint._base(np.array(x, float), np.array(y, float)) is None


def test_wrong_direction_k50_keeps_its_rising_curve_handling():
    x, y = RISING
    assert math.isnan(law._k50(_cell(RISING), "nodes")) and math.isnan(contrasts._k50(x, y))
    from reach.analysis.s2.confirm import _k50_signed, k50_difference
    assert math.isnan(_k50_signed(x, y, [f"t{i // 5}" for i in range(len(x))]))
    assert math.isnan(alloc._k50([3, 4, 6, 7], [0.0, 0.0, 1.0, 1.0]))
    d = k50_difference(_cell(RISING), _cell(COMPLETE), boot=20)
    assert math.isnan(d["diff"]) and not d["holds"]


def test_truly_degenerate_k50_stays_undefined():
    assert all(math.isnan(v) for v in K50.k50([3, 4, 5], [1, 1, 1], ["a", "b", "c"], boot=5))
    assert math.isnan(law._k50(_cell(([5] * 10, [1, 0] * 5)), "nodes"))
    assert math.isnan(contrasts._k50([5, 5, 5, 5], [1, 0, 1, 0]))
    assert math.isnan(alloc._k50([5, 5], [1.0, 0.0])) and alloc._k50([3, 5], [1.0, 1.0]) == INF


def _budget_records(level8):
    rng = np.random.default_rng(3)
    recs = []
    for b in (1, 2, 4, 8):
        for K in range(4, 20, 2):
            for t in range(6):
                for s in range(5):
                    if b == 8:
                        y = level8(K)
                    else:
                        y = int(rng.random() < 1 / (1 + math.exp(-(0.5 * (8 + 2 * math.log2(b) - K)))))
                    recs.append(dict(set="S", target_id=f"K{K}_{t}", seed=s, K=K, budget=b,
                                     units=dict(nodes=float(K)), y=y))
    return recs


def test_reach_per_doubling_uses_a_separated_level_at_its_gap_midpoint():
    """A budget level separated in the predicted direction enters the line at its gap midpoint (here (10 + 12) / 2);
    one separated in the wrong direction is left out, as a rising level always was."""
    by = law._by_target(_budget_records(lambda K: int(K <= 10)))
    pts = [(math.log2(b), law._k50([r for t in by for r in by[t] if r["budget"] == b], "nodes")) for b in (1, 2, 4)]
    x, y = np.array(pts + [(3.0, 11.0)]).T
    assert law._per_doubling(by, list(by), "nodes", (1, 2, 4, 8)) == pytest.approx(float(np.polyfit(x, y, 1)[0]))
    rising = law._by_target(_budget_records(lambda K: int(K >= 12)))
    assert law._per_doubling(rising, list(rising), "nodes", (1, 2, 4, 8)) == pytest.approx(
        law._per_doubling(rising, list(rising), "nodes", (1, 2, 4)))


def _fe_pool(overlap):
    """Function fixed effects: every function succeeds in its smaller-size sets and fails in its larger ones."""
    recs = []
    for f in range(8):
        for j, s in enumerate(("S0", "S1", "S2")):
            K = 4 + f % 3 + 2 * j
            for k in range(6):
                y = int(j == 0) if not (overlap and f == 0) else int(k % 2 == 0)
                recs.append(dict(set=s, target_id=f"{s}_pool{f:03d}", seed=k, K=K, units={}, budget=1, y=y))
    return recs


def test_h3_fixed_effects_refuse_joint_separation():
    """Row 118 kept (D088): H3 decides by a clustered Wald CI, which has no limit under separation."""
    with pytest.raises(ValueError, match="separat"):
        within.fixed_effects(_fe_pool(overlap=False))
    fe = within.fixed_effects(_fe_pool(overlap=True))                      # one function overlaps: beta exists
    assert np.isfinite(fe["coef"]) and fe["functions"] == 8


def _h2(separated_set):
    rng = np.random.default_rng(5)
    recs = []
    for g in ("S0", "S1", "S2", "S3"):
        for K in range(3, 30, 2):
            for t in range(3):
                for s in range(5):
                    if g == separated_set:
                        y = int(K < 15)
                    else:
                        y = int(rng.random() < 1 / (1 + math.exp(-(4 - 0.3 * K))))
                    recs.append(dict(set=g, target_id=f"{g}_{K}_{t}", y=y, units=dict(nodes=float(K))))
    return recs


def test_h2_a_separated_within_set_fit_is_not_estimable():
    """Row 118 kept (D088): H2's within-set cross-validation fits a logistic per fold; the limit of a separated fold
    fit predicts 0 and 1, so its log-loss would be set by the clip, not the data. The excess is not estimable and
    replicates with such a fit are dropped."""
    r = invariance.invariance_test(_h2("S0"), ["nodes"], delta=0.02, boot=20, seed=3)["nodes"]
    assert math.isnan(r["excess"]) and not r["estimable"] and not r["invariant"] and r["dropped"] == 20
    assert math.isnan(r["p_below_delta"]) and r["upper95"] == float("inf")
    ps = invariance.per_set_test(_h2("S0"), "nodes", boot=20, seed=3)
    assert not ps["S0"]["estimable"] and ps["S0"]["upper95"] == float("inf")
    ok = invariance.invariance_test(_h2(None), ["nodes"], delta=0.02, boot=20, seed=3)["nodes"]
    assert "dropped" not in ok and "estimable" not in ok and np.isfinite(ok["excess"])   # unchanged without separation


def test_separated_columns_finds_the_coefficients_without_a_finite_value():
    x, y = np.array(COMPLETE[0], float), np.array(COMPLETE[1], float)
    assert K50.separated_columns(np.column_stack([np.ones(len(x)), x]), y) == [0, 1]
    x2 = np.array([3, 4, 5, 6] * 5, float)
    assert K50.separated_columns(np.column_stack([np.ones(20), x2]), np.array([1, 0, 1, 0] * 5, float)) == []
    # one set all failures: only its intercept diverges; the slope keeps a finite value
    X = np.array([[1, 0, 3], [1, 0, 5], [1, 0, 3], [1, 0, 5], [0, 1, 3], [0, 1, 5]], float)
    assert K50.separated_columns(X, np.array([1, 0, 0, 1, 0, 0], float)) == [1]
    # two sets, each separated the same way: the common slope diverges
    X = np.array([[1, 0, 3], [1, 0, 5], [1, 0, 4], [1, 0, 5], [0, 1, 3], [0, 1, 5], [0, 1, 4]], float)
    assert 2 in K50.separated_columns(X, np.array([1, 0, 1, 1, 1, 0, 0], float))
    # one set separated by size, the other overlapping: every coefficient keeps a finite value (D088: the multi-
    # predictor fits already use such a set; only a joint separation is refused)
    X = np.array([[1, 0, 3], [1, 0, 5], [0, 1, 3], [0, 1, 5], [0, 1, 3], [0, 1, 5]], float)
    assert K50.separated_columns(X, np.array([1, 0, 1, 0, 0, 1], float)) == []


def _law_records(separate):
    rng = np.random.default_rng(2)
    recs = []
    for g in ("S0", "S1"):
        for K in range(4, 22, 2):
            for t in range(4):
                for s in range(4):
                    for b in (1, 2, 4, 8, 16):
                        eta = 0.6 * (12 + 1.5 * math.log2(b) - K)
                        y = int(eta > 0) if separate else int(rng.random() < 1 / (1 + math.exp(-eta)))
                        recs.append(dict(set=g, target_id=f"{g}_K{K}_{t}", seed=s, K=K, budget=b,
                                         units=dict(nodes=float(K)), y=y))
    return recs


@pytest.mark.filterwarnings("ignore")
def test_pooled_law_and_curvature_refuse_separated_slopes():
    """Row 118 kept (D088): gamma/beta has no unique limit and the clustered SEs have none."""
    for ss in (False, True):
        f = law.fit_law(_law_records(True), "nodes", set_specific=ss)
        assert math.isnan(f["beta"]) and math.isnan(f["gamma"]) and math.isnan(f["rate"])
        c = law.curvature(_law_records(True), "nodes", set_specific=ss)
        assert math.isnan(c["quad"]) and math.isnan(c["p_wald"])
        r = law.rate_ci(_law_records(True), "nodes", boot=20, set_specific=ss)
        assert not r["estimable"] and r["dropped"] == 20               # Newton steps never converge: dropped
    f = law.fit_law(_law_records(False), "nodes", set_specific=True)
    assert f["beta"] == pytest.approx(0.6, rel=0.25) and np.isfinite(law.curvature(_law_records(False), "nodes")["p_wald"])


def _h7_records(separate):
    rng = np.random.default_rng(4)
    recs = []
    for g in ("S0", "S1", "S2"):
        for K in range(4, 20, 2):
            for t in range(3):
                for s in range(4):
                    for b in (1, 4):
                        eta = 0.5 * (11 + math.log2(b) - K)
                        y = int(eta > 0) if separate else int(rng.random() < 1 / (1 + math.exp(-eta)))
                        recs.append(dict(set=g, target_id=f"{g}_{K}_{t}", K=K, budget=b, y=y,
                                         units=dict(nodes=float(K)), features=dict(n=1.0)))
    return recs


@pytest.mark.filterwarnings("ignore")
def test_h7_predictor_never_selects_or_freezes_a_separated_fit():
    """Row 118 kept (D088): H7 selects by log-loss and freezes finite coefficients."""
    from reach.analysis.s2 import predictor
    with pytest.raises(ValueError, match="separated"):
        predictor.select(_h7_records(True), ["nodes"])
    assert np.isfinite(predictor.select(_h7_records(False), ["nodes"])["loso_logloss"])
    assert all(math.isnan(v) for v in predictor._calibration(np.array([1, 1, 0, 0.]), np.array([.9, .8, .2, .1])))
    recs = _h7_records(False)
    p = np.array([1.0 if r["y"] else 0.0 for r in recs])                # perfect predictions: separated
    res = predictor.score_p(recs, np.clip(p, 0.01, 0.99), np.full(len(p), 0.5), eps_set=1.0, boot=20,
                              min_sets_ok=len({r["set"] for r in recs}))
    assert res["intercept_dropped"] == 20 and math.isnan(res["intercept_ci"][0]) and not res["passes"]


@pytest.mark.filterwarnings("ignore")
def test_h10b_transfer_fit_refuses_separated_data():
    """Row 118 applied to H10b's one-predictor recalibration (D088; D086's open item): its limit is a 0/1 step on eta,
    not a model of the frozen form with finite c0 and c1, so it is refused, never frozen with large finite values."""
    from reach.analysis.s2 import cgp_confirm as C
    frozen = dict(unit="nodes", budget_term=True, feature=None, params=[4.0, -0.5, 0.6], loso_logloss=0.3,
                  sha256="f" * 64)
    recs = []
    for g in ("S0", "S1"):
        for K in (3, 5, 7, 9, 11, 13):
            for i in range(4):
                for b in C.FIT_BUDGETS:
                    eta = 4.0 - 0.5 * K + 0.6 * math.log2(b)
                    recs.append(dict(set=g, target_id=f"{g}_{K}_{i}", seed=0, algorithm="CGP", K=K,
                                     units=dict(nodes=float(K)), budget=b, y=int(eta > 0)))
    with pytest.raises(ValueError, match="separated"):
        C.fit_transfer(frozen, recs)


def test_separation_counts_per_family_in_the_report():
    """D088: the report states how often a fit was separated, one short line per family: slope families from the
    cells (point fits and replicates), K50 families from the one-predictor fits counted under their label."""
    from reach.analysis.s2 import confirm
    K50.TALLY.clear()
    cells = [dict(set="BOOL-NAND-n5", separated="falling", separated_reps=150, dropped=2),
             dict(set="BOOL-NAND-n5", separated_reps=3, dropped=0), dict(set="R-ARITH4", dropped=1)]
    with K50.counting("H5"):
        K50.fit1(*COMPLETE)
        K50.fit1(*RISING)
        K50.fit1([3, 4, 5, 6], [1, 0, 1, 0])
        K50.fit1([3, 4], [1, 1])                                                     # degenerate: not a fit
    K50.fit1(*COMPLETE)                                                              # no label: not counted
    s = confirm.separation_summary(dict(h1=cells, h8=[], h9a=[]), boot=200, tally=K50.TALLY)
    K50.TALLY.clear()
    assert s["H1"] == dict(cells=3, separated_cells=1, falling=1, rising=0, replicates=600, separated_replicates=153)
    assert s["H5"] == dict(fits=3, separated=2, falling=1, rising=1)
    lines = confirm.separation_lines(dict(separation=s))
    assert any(ln.startswith("- **H1:** 1 of 3 cells separated") and "153 of 600 replicates" in ln for ln in lines)
    assert any(ln.startswith("- **H5:** 2 of 3 one-predictor K₅₀ fits separated") for ln in lines)


def test_h10a_reports_its_separation_counts():
    """H10a (Stage 2) uses the H1 code; its results carry the same count (D088), in both of its families."""
    from reach.analysis.s2 import cgp_confirm as C
    from test_cgp_confirm import _rows
    rows = _rows("F-EML", 3.0, -0.5, levels=(5, 9, 13), n_targets=6, seeds=2)
    res = C.h10a(rows, boot=20, declared=None)
    assert res["separation"] == law.separation_count(res["cells"], 20) and res["separation"]["cells"] == len(res["cells"])
    cells = [dict(separated="rising", separated_reps=4), dict(separated_reps=1), dict()]
    assert law.separation_count(cells, 10) == dict(cells=3, separated_cells=1, falling=0, rising=1, replicates=30,
                                                    separated_replicates=5)


def test_dryrun_reports_the_separation_counts(tmp_path):
    from reach.analysis.s2 import dryrun as D
    from test_dryrun import SETS, _rows, _write
    _write(tmp_path, "s2_pilot_reach_a", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)])
    _write(tmp_path, "s2_pilot_mech", _rows("BOOL-NAND-n5", "B", 0.5, 9))
    res = D.build(tmp_path, delta=0.01, boot=20)
    assert set(res["separation"]) >= {"H1", "H8a", "H9a", "H4 per set, H4c", "H6", "H5"}
    assert res["separation"]["H1"]["cells"] == len(res["h1"])
    text = D.render(res)
    assert "## Separated fits (D088)" in text and "- **H1:**" in text

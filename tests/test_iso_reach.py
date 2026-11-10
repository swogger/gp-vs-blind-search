"""Isotonic interpolated reach (REQ-ANA-108; Stage 5 Rule S2): hand-computed cases, monotonicity in successes, the
bracketing property where the logistic K50 moves, ties and empty levels."""
import math

import numpy as np

from reach.analysis.s2 import iso, law


def test_hand_computed_cases():
    r = iso.reach([1, 2, 3, 4], [1.0, 0.8, 0.3, 0.0])
    assert r["flag"] == "bracketed" and (r["lo"], r["hi"]) == (2.0, 3.0)
    assert math.isclose(r["reach"], 2 + 0.3 / 0.5)                       # 2.6
    # a violation pooled with equal weights: 0.4 and 0.6 become 0.5 and 0.5; the last level at 1/2 is 3
    r = iso.reach([1, 2, 3, 4], [0.9, 0.4, 0.6, 0.1])
    assert np.allclose(r["fitted"], [0.9, 0.5, 0.5, 0.1]) and math.isclose(r["reach"], 3.0)
    # weighted by targets: three targets at level 2 (0.4 each), one at level 3 (0.6): pooled 0.45
    r = iso.reach([1, 2, 2, 2, 3, 4], [0.9, 0.4, 0.4, 0.4, 0.6, 0.1])
    assert np.allclose(r["fitted"], [0.9, 0.45, 0.45, 0.1])
    assert math.isclose(r["reach"], 1 + 0.4 / 0.45)
    top = iso.reach([1, 2, 3], [1.0, 0.9, 0.6])
    assert top["flag"] == "at_least_top" and top["reach"] == 3.0
    low = iso.reach([1, 2, 3], [0.4, 0.2, 0.0])
    assert low["flag"] == "at_most_lowest" and low["reach"] == 1.0
    # a rising violation at the low end pools down below 1/2: not crossing, at most the lowest level
    assert iso.reach([1, 2], [0.3, 0.6])["flag"] == "at_most_lowest"


def test_pav_fit_is_non_increasing_and_least_squares():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 7))
        y, w = rng.random(n), rng.integers(1, 5, n).astype(float)
        f = iso.pav_nonincreasing(y, w)
        assert np.all(np.diff(f) <= 1e-12)
        # least squares: no random non-increasing vector has a smaller weighted SSE
        sse = np.sum(w * (y - f) ** 2)
        for _ in range(20):
            g = np.sort(rng.random(n))[::-1]
            assert sse <= np.sum(w * (y - g) ** 2) + 1e-12


def _records(rates, runs=10, start=5.0, step=1.0):
    """One level per rate, 3 targets per level, `runs` runs per target, the successes spread over the targets."""
    out = []
    for i, p in enumerate(rates):
        for t in range(3):
            k = int(round(p * runs))
            out += [dict(target_id=f"L{i}t{t}", y=int(j < k), units=dict(bits_sem=start + i * step))
                    for j in range(runs)]
    return out


def _raw_first_crossing(sizes, probs):
    """Planted violation: the raw level means without PAV, read at their first downward crossing of 1/2."""
    x, m, _ = iso.levels(sizes, probs)
    for i in range(len(x) - 1):
        if m[i] >= 0.5 > m[i + 1]:
            return x[i] + (m[i] - 0.5) / (m[i] - m[i + 1]) * (x[i + 1] - x[i])
    return x[-1] if m[0] >= 0.5 else x[0]


def _monotone_violations(estimate, trials=300):
    rng, bad = np.random.default_rng(20261108), 0
    for _ in range(trials):
        n_lv = int(rng.integers(2, 8))
        recs = [dict(target_id=f"t{i}", y=int(rng.random() < 0.5), units=dict(bits_sem=float(rng.integers(0, n_lv))))
                for i in range(int(rng.integers(3, 40)))]
        before = estimate(*iso.target_probs(recs))
        fails = [i for i, r in enumerate(recs) if r["y"] == 0]
        if not fails:
            continue
        recs[int(rng.choice(fails))]["y"] = 1
        bad += estimate(*iso.target_probs(recs)) < before - 1e-12
    return bad


def test_adding_a_success_never_lowers_the_reach():
    assert _monotone_violations(lambda s, p: iso.reach(s, p)["reach"]) == 0
    assert _monotone_violations(_raw_first_crossing) > 0          # the check fires on a non-monotone estimator


def test_far_levels_keep_the_reach_between_the_bracketing_levels_while_the_logistic_moves():
    base = [1.0, 0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.0, 0.0]          # levels 5..14 bits; crossing in [9, 10]
    r0 = iso.reach_records(_records(base))
    assert r0["flag"] == "bracketed" and (r0["lo"], r0["hi"]) == (9.0, 10.0)
    k0 = law._k50(_records(base), "bits_sem")
    # far levels above the crossing with rates below 1/2 (the tolerance successes at hard levels, audit D7)
    far = iso.reach_records(_records(base + [0.3] * 6))
    k_far = law._k50(_records(base + [0.3] * 6), "bits_sem")
    assert math.isclose(far["reach"], r0["reach"]) and far["flag"] == "bracketed"
    assert abs(k_far - k0) > 1.0 and k_far > 10.0                         # the logistic leaves the bracket
    # the low levels removed (all at or above 1/2): the isotonic reach does not move at all; the logistic does
    cut = iso.reach_records(_records(base[3:], start=8.0))
    assert math.isclose(cut["reach"], r0["reach"])
    assert abs(law._k50(_records(base[3:], start=8.0), "bits_sem") - k0) > 0.1
    # planted violation of the condition: a far level with a rate above 1/2 may pull the reach out of the bracket
    pulled = iso.reach_records(_records(base + [0.0] * 6 + [1.0] * 20))
    assert pulled["reach"] > 10.0


def test_ties_and_empty_levels():
    # two sizes equal up to float noise form one level; NaN probabilities are left out
    r = iso.reach([7.0, 7.0 + 1e-12, 9.0, 11.0, 11.0], [1.0, 0.0, 0.25, float("nan"), float("nan")])
    assert r["n_levels"] == 2 and r["levels"] == [7.0, 9.0]
    assert math.isclose(r["reach"], 7.0)                                 # the mean at 7 is exactly 1/2
    # a plateau at exactly 1/2: the reach is the plateau's last level
    r = iso.reach([1, 2, 3, 4], [0.8, 0.5, 0.5, 0.2])
    assert math.isclose(r["reach"], 3.0) and r["flag"] == "bracketed"
    e = iso.reach([], [])
    assert e["flag"] == "empty" and math.isnan(e["reach"])
    assert iso.reach([3.0], [0.7])["flag"] == "at_least_top"
    assert iso.gap(iso.reach([1, 2], [0.9, 0.1]), 1.0)["gap"] == 0.5

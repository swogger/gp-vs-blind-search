"""H6d analysis (T25, D098; docs/design/h6_explanation.md version 2; REQ-ANA-87): coverage over runs that stop at a hit,
the twin and enumeration K50s by H6's own estimator, the identity gap = G - S - R + O, Stage 1's estimability rule, the
equivalence tests with Holm, and the reproduction of H6c's registered gaps. Synthetic data with known answers, plus the
registered calibration rows. Written before the code (version 2 after the independent review)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from reach.analysis.s2 import h6d as H
from reach.analysis.s2 import law

ROOT = Path(__file__).resolve().parents[1]


def _rv(snaps, stop=None):
    """snaps: {prefix: (d, undef, lv)} -> an rv record; lv_ok = lv; stop defaults to the last prefix."""
    pre = {str(p): dict(n=p, d=d, undef=u, lv=list(lv), lv_ok=list(lv)) for p, (d, u, lv) in snaps.items()}
    last = pre[str(max(snaps))]
    return dict(prefixes=pre, stop=dict(stop or last), target_first=None)


def _run(K, snaps, y=None, stop=None):
    return dict(target_id=f"t{K}", K=K, bits_sem=float(K), y=y or {}, rv=_rv(snaps, stop))


def test_pooled_actuarial_mean_uses_the_runs_still_searching():
    a = _run(1, {5000: (10, 1, [0, 2, 4]), 10000: (16, 2, [0, 3, 6]), 20000: (30, 2, [1, 5, 9])})
    b = _run(1, {5000: (20, 0, [0, 4, 2]), 10000: (26, 0, [0, 5, 4])})                    # stopped before 20,000
    c = _run(1, {5000: (6, 3, [0, 0, 2])})                                                  # stopped before 10,000
    runs = [a, b, c]
    assert H.cumulative(runs, 20000, "d", how="pooled") == pytest.approx(12 + 6 + 14)
    assert list(H.cumulative(runs, 20000, "lv", how="pooled")) == pytest.approx([1, 2 + 1 + 2, 8 / 3 + 2 + 3])
    assert math.isnan(H.cumulative([c], 20000, "d", how="pooled"))                         # nobody reached 10,000


def test_stratified_mean_weights_levels_and_borrows_the_pooled_increment():
    """Level 3: two runs that stop early (easy targets); level 9: one run that searches on. Stratified: each level's
    own life table, weighted by its share of runs; level 3's empty intervals take the pooled increment."""
    e1 = _run(3, {5000: (10, 0, [0])}, stop=dict(n=6000, d=11, undef=0, lv=[0], lv_ok=[0]))
    e2 = _run(3, {5000: (12, 0, [0])}, stop=dict(n=7000, d=13, undef=0, lv=[0], lv_ok=[0]))
    h = _run(9, {5000: (40, 0, [0]), 10000: (70, 0, [0]), 20000: (100, 0, [0])})
    runs = [e1, e2, h]
    # level 3: 11 at 5,000, then pooled increments 30 and 30 (only h is left) -> 71; level 9: 100. Weights 2/3, 1/3.
    assert H.cumulative(runs, 20000, "d", how="stratified") == pytest.approx(2 / 3 * 71 + 1 / 3 * 100)
    # carry forward: a stopped run keeps its last counts (a lower bound)
    assert H.cumulative(runs, 20000, "d", how="carry") == pytest.approx((11 + 13 + 100) / 3)
    # never hit within E: only runs that reached E with y = 0
    h["y"] = {20000: 0}
    assert H.cumulative(runs, 20000, "d", how="never_hit") == pytest.approx(100)


def test_probabilities_for_the_twin_and_for_enumeration():
    U = {1: 2, 2: 4, 3: 8, 4: 16}
    assert H.twin_probs(np.array([0, 1, 2, 4, 4]), U) == {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.25}
    assert H.enum_probs(10, U) == {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.0}                          # 2 + 4 + 4 of 8
    assert H.enum_probs(100, U) == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}


def test_twin_k50_is_h6s_estimator_on_rates_and_the_interpolation_sees_every_level():
    rng = np.random.default_rng(1)
    ks = rng.integers(3, 12, 600)
    probs = {k: 1 / (1 + math.exp(k - 7.0)) for k in range(1, 15)}
    runs = [dict(K=int(k), bits_sem=float(k)) for k in ks]
    recs = [dict(units=dict(bits_sem=float(k)), y=probs[int(k)]) for k in ks]
    assert H.k50_rates(runs, probs) == pytest.approx(law._k50(recs, "bits_sem"))
    assert H.k50_rates(runs, probs) == pytest.approx(7.0, abs=1e-6)
    U = {k: 2 ** k for k in range(1, 15)}
    bits = {k: math.log2(sum(U[j] for j in U if j <= k)) for k in U}
    step = H.enum_probs(2 ** 10, U)                                         # enumeration of 2^10 functions
    assert abs(H.k50_interp(step, U) - 10) <= max(bits[k + 1] - bits[k] for k in range(1, 14))


def _synthetic_set(k50_gp=11.0, cov_mid=8.0, n_t=40, seed=0, kmax=12, budgets=(20000,)):
    """Levels 1-kmax, U_K = 2^K, bits = log2 U(<=K); GP succeeds by a logistic in bits with midpoint k50_gp; every run
    covers level K with share 1 / (1 + exp(bits - cov_mid)) at E0 (coverage grows with budget as sqrt)."""
    rng = np.random.default_rng(seed)
    U = {k: 2 ** k for k in range(1, kmax + 1)}
    bits = {k: math.log2(sum(U[j] for j in U if j <= k)) for k in U}
    runs = []
    for i in range(n_t):
        k = int(rng.integers(3, kmax + 1))
        for s in range(5):
            y = {b: int(rng.random() < 1 / (1 + math.exp(1.2 * (bits[k] - k50_gp - 0.0 * math.log2(b / 20000)))))
                 for b in budgets}
            snaps = {}
            for p in sorted({5000, 10000, 20000} | set(budgets)):
                f = math.sqrt(p / 20000)
                lv = [0] + [min(U[j], round(f * U[j] / (1 + math.exp(bits[j] - cov_mid)))) for j in U]
                snaps[p] = (sum(lv) + 7, 0, lv)
            runs.append(dict(target_id=f"t{i}", K=k, bits_sem=bits[k], y=y, rv=_rv(snaps)))
    return runs, U


def test_terms_add_up_to_the_gap_and_spread_is_zero_for_enumeration():
    runs, U = _synthetic_set()
    t = H.set_terms(runs, 20000, U)
    assert t["G"] - t["S"] - t["R"] + t["O"] == pytest.approx(t["gap"], abs=1e-12)
    assert t["gap"] == pytest.approx(t["k50_gp"] - math.log2(20000))
    assert t["R"] == pytest.approx(math.log2(20000) - math.log2(H.cumulative(runs, 20000, "d")))
    assert abs(t["k50_twin"] - 8.0) < 0.7 and abs(t["k50_gp"] - 11.0) < 0.8 and t["G"] > 2
    # a GP whose coverage IS size-ordered enumeration of its distinct functions has no size spread
    d = H.cumulative(runs, 20000, "d")
    enum = H.enum_probs(d, U)
    for r in runs:
        lv = [0] + [round(enum[j] * U[j]) for j in U]
        r["rv"] = _rv({5000: (d, 0, lv), 10000: (d, 0, lv), 20000: (d, 0, lv)})
    t = H.set_terms(runs, 20000, U)
    assert t["S"] == pytest.approx(0, abs=0.05)


def test_runs_from_rows_reads_the_hit_by_prefix():
    rows = [dict(target_id="a", K=5, units=dict(bits_sem=5.0), exact=True, gen_exact=199, pop_size=100, budget_E=20000),
            dict(target_id="b", K=5, units=dict(bits_sem=5.0), exact=True, gen_exact=200, pop_size=100, budget_E=320000),
            dict(target_id="c", K=5, units=dict(bits_sem=5.0), exact=False, gen_exact=None, pop_size=100,
                 budget_E=320000)]
    out = H.runs_from_rows(rows, (20000, 40000))
    assert [r["y"] for r in out] == [{20000: 1}, {20000: 0, 40000: 1}, {20000: 0, 40000: 0}]
    assert all(r["rv"] is None for r in out)


def test_gap_at_e0_reproduces_h6cs_registered_estimates():
    """Review C1: E0 pools s2_cal_a0 and s2_cal_a16 at 20,000 evaluations on H6c's main sets, so the H6d gap is H6c's.
    Uses the registered calibration rows and data/analysis/calibration.json."""
    reg = json.loads((ROOT / "data/analysis/calibration.json").read_text())["h6c"]["gaps"]
    by = H.rows_by_set(ROOT, ("s2_cal_a0", "s2_cal_a16"))
    for d in ("real", "bool"):
        sets = H.main_sets(by, d)
        assert sets == reg[d]["sets"]
        gaps = {g: H.gap(H.runs_from_rows(by[g], (H.E0,)), H.E0) for g in sets}
        for g in sets:
            assert gaps[g] == pytest.approx(reg[d]["per_set"][g], abs=1e-9), g
        assert np.mean(list(gaps.values())) == pytest.approx(reg[d]["gap"], abs=1e-9)


def test_equivalence_with_holm_and_the_estimability_rule():
    """TOST by bootstrap: the 90% interval inside +-delta; Holm over the family; a test with more than 10% dropped
    replicates is not estimable and not supported (Stage 1's rule; review M1)."""
    rng = np.random.default_rng(0)
    near = rng.normal(0.1, 0.2, 2000)
    far = rng.normal(2.0, 0.2, 2000)
    a = H.equivalence(0.1, near, delta=1.0)
    b = H.equivalence(2.0, far, delta=1.0)
    assert a["equivalent"] and a["p"] < 0.025 and not b["equivalent"] and b["p"] > 0.5
    holm = H.holm({"H6d-1": a, "H6d-2": b})
    assert holm["H6d-1"]["supported"] and not holm["H6d-2"]["supported"]
    gappy = np.concatenate([near[:1500], np.full(500, np.nan)])                          # 25% dropped
    c = H.equivalence(0.1, gappy, delta=1.0)
    assert not c["estimable"] and not H.holm({"H6d-1": c})["H6d-1"]["supported"]


def test_domain_tests_fire_on_planted_differences():
    """H6d-1 passes when the twins match across domains (the gap difference is guidance) and fails when the domains
    differ in coverage; H6d-2 passes when real guidance does not change with budget."""
    def dom(k50s, mids, seed, kmax=12):
        return {f"S{i}": _synthetic_set(k50_gp=k, cov_mid=m, seed=seed + i, kmax=kmax, budgets=(20000, 40000, 80000))
                for i, (k, m) in enumerate(zip(k50s, mids))}
    real = dom([9.5, 9.0, 9.8], [7.0, 7.0, 7.0], 10)
    same = dom([20.0, 21.0, 19.5], [7.0, 7.0, 7.0], 30, kmax=26)      # same twin: the difference is guidance
    other = dom([12.0, 12.5, 11.8], [10.0, 10.2, 9.9], 40, kmax=16)    # twin 3 bits higher: coverage differs
    res = H.domain_tests(dict(real=real, bool=same), budgets=(20000, 40000, 80000), boot=200, seed=5)
    assert res["H6d-1"]["supported"], res["H6d-1"]
    res2 = H.domain_tests(dict(real=real, bool=other), budgets=(20000, 40000, 80000), boot=200, seed=5)
    assert not res2["H6d-1"]["supported"]
    assert set(res["per_set"]["real"]["S0"]) >= {"gap", "R", "S", "G", "O"}


def test_power_and_the_margin_for_80_percent():
    """Power of a bootstrap TOST at the registered sample size, from a pilot's estimate and its replicates scaled by
    sqrt(n_pilot / n_registered); the margin giving 80% power is |estimate| + (z_0.95 + z_0.80) x SE."""
    rng = np.random.default_rng(1)
    reps = rng.normal(0.2, 0.4, 4000)                                       # pilot SE 0.4
    pw = H.power(0.2, reps, delta=1.0, n_pilot=1, n_registered=4)          # registered SE 0.2
    assert pw["se"] == pytest.approx(0.2, rel=0.05)
    assert pw["power"] == pytest.approx(0.993, abs=0.01)                    # Phi((1 - 0.2) / 0.2 - 1.645)
    assert pw["delta_80"] == pytest.approx(0.2 + (1.6449 + 0.8416) * 0.2, rel=0.05)


def test_secondary_margins_are_their_own_family():
    """Secondary tests reuse the primary statistics' replicates at other margins, with Holm over the secondary family
    only; the primary decisions are unchanged by them."""
    def dom(k50s, mids, seed, kmax=12):
        return {f"S{i}": _synthetic_set(k50_gp=k, cov_mid=m, seed=seed + i, kmax=kmax, budgets=(20000, 40000, 80000))
                for i, (k, m) in enumerate(zip(k50s, mids))}
    real = dom([9.5, 9.0, 9.8], [7.0, 7.0, 7.0], 10)
    other = dom([12.0, 12.5, 11.8], [10.0, 10.2, 9.9], 40, kmax=16)        # twins about 3 bits apart
    a = H.domain_tests(dict(real=real, bool=other), budgets=(20000, 40000, 80000), boot=200, seed=5)
    b = H.domain_tests(dict(real=real, bool=other), budgets=(20000, 40000, 80000), boot=200, seed=5,
                       secondary={"H6d-1": 5.0, "H6d-2": 2.0})
    assert a["H6d-1"] == b["H6d-1"] and not b["H6d-1"]["supported"]
    assert b["secondary"]["H6d-1"]["delta"] == 5.0 and b["secondary"]["H6d-1"]["supported"]

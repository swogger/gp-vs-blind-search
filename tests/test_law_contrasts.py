"""T16 pooled law and contrasts (written before the code). REQ-ANA-07, REQ-RULE-05."""
import math

import numpy as np
import pytest

from reach.analysis.s2 import contrasts as C
from reach.analysis.s2 import law as L


def _sim(beta=0.6, gamma=0.4, alpha=6.0, sd_target=1.0, seed=1):
    rng = np.random.default_rng(seed)
    recs = []
    for K in range(4, 24, 2):
        for t in range(15):
            u = rng.normal(0, sd_target)                                  # target effect: clustering
            for s in range(5):
                for b in (1, 2, 4, 8, 16):
                    eta = alpha - beta * K + gamma * math.log2(b) + u
                    recs.append(dict(set="S", target_id=f"K{K}_{t}", seed=s, K=K, budget=b, units=dict(nodes=float(K)),
                                     y=int(rng.random() < 1 / (1 + math.exp(-eta)))))
    return recs


def test_pooled_law_recovers_rate_and_clusters():
    fit = L.fit_law(_sim(sd_target=0.0), unit="nodes")
    assert fit["beta"] == pytest.approx(0.6, rel=0.12) and fit["gamma"] == pytest.approx(0.4, rel=0.15)
    assert fit["rate"] == pytest.approx(0.4 / 0.6, rel=0.2)                 # nodes per doubling
    clustered = L.fit_law(_sim(sd_target=1.5), unit="nodes")
    assert clustered["se_beta_cluster"] > clustered["se_beta_naive"]       # clustering widens the SE


def test_holm_counts_untestable_members():
    """Lesson 16: the family size includes members with no test; Holm uses m, not the number of p-values."""
    adj = C.holm([0.01, 0.04, None, None], m=4)
    assert adj[0] == pytest.approx(0.04) and adj[1] == pytest.approx(0.12) and adj[2] is None
    assert C.holm([0.01, 0.04], m=2)[1] == pytest.approx(0.04)


def test_mcnemar_exact():
    # 10 discordant pairs, 9 favour A: two-sided exact p = 2 * P(X <= 1 | Bin(10, 0.5)) = 22/1024
    p = C.mcnemar_exact(9, 1)
    assert p == pytest.approx(22 / 1024)
    assert C.mcnemar_exact(0, 0) is None                                    # untestable


def test_paired_k50_difference_covers_the_truth():
    rng = np.random.default_rng(4)
    recs = []
    for K in range(4, 26, 2):
        for t in range(12):
            for s in range(4):
                for algo, shift in (("A", 2.0), ("B", 0.0)):
                    p = 1 / (1 + math.exp(-(0.7 * (12 + shift - K))))
                    recs.append(dict(algorithm=algo, target_id=f"K{K}_{t}", seed=s, K=K, y=int(rng.random() < p)))
    d = C.paired_k50_difference(recs, "A", "B", boot=200, seed=5)
    assert d["lo"] <= 2.0 <= d["hi"] and d["diff"] == pytest.approx(2.0, abs=0.6)


def test_h1_slope_ci_excludes_zero_only_when_size_matters():
    """H1 per cell: target-bootstrap CI of the logistic slope (positive and negative control)."""
    cell = [r for r in _sim() if r["budget"] == 1]
    s = L.slope_ci(cell, boot=300)
    assert s["hi"] < 0 and s["slope"] == pytest.approx(-0.6, rel=0.3)
    covered = 0
    for k in range(20):                                       # a 95% CI on null data covers 0 about 19 times in 20
        rng = np.random.default_rng(100 + k)
        f = L.slope_ci([dict(r, y=int(rng.random() < 0.5)) for r in cell], boot=200, seed=k)
        covered += f["lo"] < 0 < f["hi"]
    assert covered >= 16


def test_reach_per_doubling_ci_covers_gamma_over_beta():
    """H4, H6a: slope of K50 on log2 budget, with targets resampled jointly across budgets."""
    r = L.reach_per_doubling(_sim(sd_target=0.5), unit="nodes", budgets=(1, 2, 4, 8, 16), boot=200)
    assert r["lo"] < 0.4 / 0.6 < r["hi"] and r["hi"] - r["lo"] < 0.6


def test_curvature_detects_a_quadratic_budget_term_only_when_present():
    """H4a: reach is linear in log2 budget; the quadratic coefficient's clustered CI covers 0 on linear data."""
    lin = L.curvature(_sim(sd_target=0.5), unit="nodes")
    assert lin["lo"] < 0 < lin["hi"]
    rng = np.random.default_rng(3)
    bent = [dict(r, y=int(rng.random() < 1 / (1 + math.exp(-(6 - 0.6 * r["K"] + 1.2 * math.log2(r["budget"])
                                                                - 0.25 * math.log2(r["budget"]) ** 2)))))
            for r in _sim()]
    c = L.curvature(bent, unit="nodes")
    assert c["hi"] < 0


def test_saturated_contrast_is_reported_not_a_crash():
    """Review finding 5: when one algorithm never succeeds, every replicate is NaN; the contrast is not estimable."""
    recs = [dict(target_id=f"t{K}_{i}", algorithm=a, K=K, y=(1 if a == "A" and K < 10 else 0))
            for K in range(4, 20, 2) for i in range(5) for a in ("A", "B")]
    r = C.paired_k50_difference(recs, "A", "B", boot=50)
    assert not r["estimable"] and math.isnan(r["lo"]) and math.isnan(r["hi"])

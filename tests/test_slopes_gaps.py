"""H4c, H6b and H6c as filed in Stage 1 (REQ-ANA-49 to 51; D081 a-c; guardrail row 109), written before the code.
Synthetic data only.

Stage 1 (OSF a3spg), as filed: "H4c: per-set slopes (bits_sem per doubling) are equal within each domain within +-0.5
(real) or +-1.0 (Boolean)"; "H6b: the GP-null gap changes with size (size x method interaction). H6c (secondary): the
gap at E0 has opposite signs in the two domains."; "Directional effects: the 95% CI excludes 0 in the predicted
direction. Equivalence: the 95% CI lies within the margin."; "P-values for Holm come from the bootstrap by CI inversion."
"""
import math

import numpy as np
import pytest

from reach.analysis.s2 import law as L
from reach.analysis.s2 import nullgap as N
from reach.analysis.s2.load import E0


def _budget_set(g, rate, k0=10.0, b1=1.0, seed=0, levels=(6, 8, 10, 12, 14, 16, 18, 20), targets=30, runs=5,
                budgets=(1, 2, 4, 8, 16)):
    """Records whose K50 (bits_sem = K) moves by `rate` per budget doubling: P = sigmoid(-b1 (K - k0 - rate log2 b))."""
    rng = np.random.default_rng(seed)
    out = []
    for K in levels:
        for t in range(targets):
            for b in budgets:
                p = 1 / (1 + math.exp(b1 * (K - k0 - rate * math.log2(b))))
                for s in range(runs):
                    out.append(dict(set=g, target_id=f"{g}_K{K}_{t}", seed=s, K=K, K_proven=K, budget=b,
                                    y=int(rng.random() < p), units=dict(nodes=float(K), bits_sem=float(K))))
    return out


BUD = (1, 2, 4, 8, 16)


def test_h4c_pairs_equivalence_known_answer_and_conjunction():
    """Every pair of per-set slopes within +-margin (two-sided 95% CI); member p = the largest pair p (intersection-
    union); pair p < 0.05 exactly when its CI lies inside (-m, m)."""
    band = {g: _budget_set(g, r, seed=i) for i, (g, r) in enumerate((("A", 1.0), ("B", 1.1), ("C", 1.05)))}
    h = L.slope_equality(band, {g: BUD for g in band}, margin=0.5, boot=200)
    assert len(h["pairs"]) == 3 and h["untestable"] is None and h["unit"] == "bits_sem"
    for g, r in (("A", 1.0), ("B", 1.1), ("C", 1.05)):
        assert h["sets"][g]["rate"] == pytest.approx(r, abs=0.2)
    for pr in h["pairs"]:
        assert (pr["p"] < 0.05) == (-0.5 < pr["lo"] and pr["hi"] < 0.5)
        assert pr["diff"] == pytest.approx(h["sets"][pr["sets"][0]]["rate"] - h["sets"][pr["sets"][1]]["rate"])
    assert h["p"] == max(pr["p"] for pr in h["pairs"]) and h["p"] < 0.05 and h["holds"]
    # one set off by a whole bit per doubling: that pair fails, and so does the member (largest p, not the smallest)
    band["D"] = _budget_set("D", 2.0, seed=9)
    h = L.slope_equality(band, {g: BUD for g in band}, margin=0.5, boot=200)
    bad = [pr for pr in h["pairs"] if "D" in pr["sets"]]
    assert len(h["pairs"]) == 6 and all(pr["p"] > 0.05 and not pr["holds"] for pr in bad)
    assert h["p"] == max(pr["p"] for pr in h["pairs"]) > 0.05 and not h["holds"]


def test_h4c_compares_pairs_not_each_set_with_the_mean():
    """Rates 0, 0.4 and 0.8 with margin 0.5: each set lies within 0.5 of the mean, but the extreme pair differs by 0.8.
    H4c ("equal within +-0.5") is not supported (the pooled-mean reading would support it)."""
    band = {g: _budget_set(g, r, seed=20 + i) for i, (g, r) in enumerate((("A", 0.0), ("B", 0.4), ("C", 0.8)))}
    h = L.slope_equality(band, {g: BUD for g in band}, margin=0.5, boot=200)
    ext = next(pr for pr in h["pairs"] if set(pr["sets"]) == {"A", "C"})
    assert ext["diff"] == pytest.approx(-0.8, abs=0.2) and not ext["holds"] and h["p"] >= ext["p"] > 0.05


def test_h4c_untestable_with_fewer_than_two_sets_or_a_degenerate_set():
    one = {"A": _budget_set("A", 1.0)}
    assert L.slope_equality(one, {"A": BUD}, margin=0.5, boot=20)["untestable"] == "fewer than 2 sets"
    flat = [dict(r, y=1) for r in _budget_set("F", 1.0, seed=3)]                 # every run exact: no K50
    band = {"A": _budget_set("A", 1.0), "F": flat}
    h = L.slope_equality(band, {g: BUD for g in band}, margin=0.5, boot=20)
    assert h["p"] is None and h["untestable"].startswith("not estimable") and not h["holds"]


# ------------------------------------------------------------------ H6b: the GP-null gap against size

def _levels(U, ceiling_cum=None):
    """Enumeration counts: {K: U(K)} -> counts.json-style levels with U_cum."""
    out, c = [], 0
    for K in sorted(U):
        c += U[K]
        out.append(dict(K=K, U=U[K], U_cum=c, candidates=U[K]))
    return out


def test_null_hit_probability_from_counts_and_above_the_ceiling():
    lv = _levels({1: 2, 3: 8, 5: 30, 7: 160})                                  # U(<=5) = 40, U(<=7) = 200
    assert N.null_p(lv, 5, True, 25) == pytest.approx((25 - 10) / 30)
    assert N.null_p(lv, 3, True, 25) == 1.0 and N.null_p(lv, 7, True, 25) == 0.0
    assert N.null_p(lv, 9, False, 150) == 0.0                                  # constructed: above the ceiling
    assert math.isnan(N.null_p(lv, 9, False, 250))                             # U(<=ceiling) < E: not computable


def _gap_records(g, counts, shift):
    """4 runs per target, 3 targets per level; `counts[K]` + shift successes per target at level K."""
    out = []
    for K, k in counts.items():
        for t in range(3):
            for s in range(4):
                out.append(dict(set=g, target_id=f"{g}_{K}_{t}", seed=s, K=K, K_proven=K, budget=1,
                                y=int(s < min(4, k + shift)), units=dict(nodes=float(K), bits_sem=float(K))))
    return out


def test_h6b_gap_slope_known_answer_with_set_intercepts():
    """Per target gap = mean GP recovery - the primary null's hit probability at E0; slope on size (nodes) with an
    intercept per set (the size x method interaction of a pooled linear model). Known answer computed directly; the
    sets have different sizes and gap levels, so a slope without set intercepts differs."""
    counts = {5: 10, 7: 15000, 9: 20000, 11: 10 ** 6, 13: 10 ** 7}
    lv = {"A": _levels(counts), "B": _levels(counts)}
    ka, kb = {5: 4, 7: 3, 9: 2, 11: 0}, {9: 4, 11: 4, 13: 3}
    recs = {"A": _gap_records("A", ka, 0), "B": _gap_records("B", kb, 0)}
    h = N.h6b(recs, lv, E0, boot=100)
    pn = {5: 1.0, 7: 1.0, 9: (E0 - 15010) / 20000, 11: 0.0, 13: 0.0}
    xs, ys = [], []
    for k in (ka, kb):
        x = np.array([K for K in k for _ in range(3)], float)
        y = np.array([k[K] / 4 - pn[K] for K in k for _ in range(3)])
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    assert h["theta"] == pytest.approx(float((x * y).sum() / (x * x).sum()), rel=1e-12)
    xa = np.array([K for k in (ka, kb) for K in k for _ in range(3)], float)
    ya = np.array([k[K] / 4 - pn[K] for k in (ka, kb) for K in k for _ in range(3)])
    assert abs(h["theta"] - np.polyfit(xa, ya, 1)[0]) > 0.02                  # one intercept would give another slope
    assert h["unit"] == "nodes" and h["targets"] == 21 and h["left_out"] == 0
    assert (h["p"] < 0.05) == (h["lo"] > 0 or h["hi"] < 0)                    # "changes": two-sided
    assert h["p"] == pytest.approx(min(L.p_bound(h["_reps"], 0, "below", 2), L.p_bound(h["_reps"], 0, "above", 2)))
    lvA = {row[0]: row[1] for row in h["per_set"]["A"]["levels"]}
    assert lvA[9] == pytest.approx(0.5 - pn[9]) and lvA[5] == 0.0
    assert h["per_set"]["A"]["sign_changes"] == [[7, 9]]                       # - to + between 7 and 9; + to 0 is not one


def test_h6b_leaves_out_targets_without_a_null_probability():
    lv = {"A": _levels({5: 10, 7: 50}), "B": _levels({5: 10, 7: 25000})}       # A's U(<=ceiling) = 60 < E0
    recs = {g: _gap_records(g, {5: 4, 7: 3, 9: 1}, 0) for g in ("A", "B")}
    for g in recs:
        for r in recs[g]:
            if r["K"] == 9:
                r["K_proven"] = None                                           # constructed size above the ceiling
    h = N.h6b(recs, lv, E0, boot=20)
    assert h["left_out"] == 3 and h["targets"] == 15


# ------------------------------------------------------------------ H6c: opposite signs in the two domains

def test_h6c_p_is_the_better_orientation_of_two_sided_inversions():
    rng = np.random.default_rng(1)
    neg, pos = rng.normal(-2, 0.5, 400), rng.normal(2, 0.5, 400)
    assert N.p_opposite(neg, pos) == 0.0 and N.p_opposite(pos, neg) == 0.0
    assert N.p_opposite(neg, neg) == 1.0
    mixed = rng.normal(-1.0, 0.55, 400)                                        # real gap near the edge
    p = N.p_opposite(mixed, pos)
    assert p == pytest.approx(L.p_bound(mixed, 0, "below", 2)) and 0 < p < 0.2  # two-sided, not one-sided
    assert N.p_same(neg, neg) == 0.0 and N.p_same(neg, pos) == 1.0
    assert N.p_same(mixed, neg) == pytest.approx(L.p_bound(mixed, 0, "below", 2))


def test_h6c_domain_gap_is_the_mean_set_gap_in_bits_sem_at_e0():
    sets = {g: [dict(r, budget=1) for r in _budget_set(g, 0.0, k0=k, b1=1.5, seed=i, budgets=(1,),
                                                       levels=tuple(range(6, 26, 2)))]
            for i, (g, k) in enumerate((("A", 12.0), ("B", 16.0)))}
    d = N.domain_gap(sets, E0, boot=100)
    assert d["gap"] == pytest.approx(14.0 - math.log2(E0), abs=0.3) and d["estimable"]
    assert set(d["per_set"]) == {"A", "B"} and len(d["_reps"]) == 100 - d["dropped"]
    real = dict(d)
    boolean = N.domain_gap({g: [dict(r, units=dict(r["units"], bits_sem=r["K"] + 6.0)) for r in rr]
                            for g, rr in sets.items()}, E0, boot=100)
    h = N.h6c({"real": real, "bool": boolean})
    assert h["p"] == pytest.approx(N.p_opposite(real["_reps"], boolean["_reps"])) and h["untestable"] is None
    assert h["orientation"] == "real below 0, Boolean above 0"
    assert N.h6c({"bool": boolean})["untestable"] == "missing: no data in real"


def test_per_set_rates_are_reported_beside_h4b_as_exploratory():
    """Review (2026-10-27): the slopes-and-gaps file reports each set's bits_sem rate per doubling with whether its
    two-sided 95% CI lies below 1, beside the pooled H4b (real domain), labelled exploratory."""
    from reach.analysis.s2 import confirm as C
    ps = lambda r, lo, hi: dict(budgets=[1, 2, 4, 8, 16], nodes=dict(rate=1.0, lo=0.5, hi=1.5),     # noqa: E731
                                bits_sem=dict(rate=r, lo=lo, hi=hi))
    res = dict(label="T", h6b={}, h4=dict(real=dict(per_set={"R-A": ps(0.6, 0.4, 0.8), "R-B": ps(0.9, 0.7, 1.1)},
                                                    rate_bits_sem=dict(rate=0.7, lo=0.6, hi=0.8, upper95=0.78),
                                                    h4c=dict(pairs=[], margin=0.5)),
                                          bool=dict(per_set={"B-A": ps(1.2, 1.1, 1.3)}, h4c=dict(pairs=[], margin=1.0))))
    text = C.render_slopes_gaps(res, "t", "s")
    assert "exploratory" in text and "H4b" in text
    rows = {ln.split(" | ")[0].strip("| "): ln for ln in text.splitlines() if ln.startswith("| R-") or ln.startswith("| B-")}
    assert rows["R-A"].rstrip().endswith("| yes |") and rows["R-B"].rstrip().endswith("| no |")
    assert rows["B-A"].rstrip().endswith("| – |")                       # H4b is filed for the real domain only
    assert "pooled real rate 0.70 bits_sem per doubling, upper 95% bound 0.78" in text

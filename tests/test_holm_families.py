"""The registered Holm families of Stage 1 other than H1 (REQ-ANA-29; guardrail row 96), written before the code.
Synthetic data only.

Stage 1 (OSF a3spg), Inference criteria: "Directional effects: the 95% CI excludes 0 in the predicted direction.
Equivalence: the 95% CI lies within the margin. Holm correction within families declared now, each counting untestable
members: per domain, H1 cells; H2 units (4); H4a-f; H5 contrasts x sets; H6a-c; H8 configurations x sets (18); H9a
configurations x sets (6). P-values for Holm come from the bootstrap by CI inversion."
"""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import families as F
from reach.analysis.s2 import law as L

# ------------------------------------------------------------------ p by inverting a registered bound

BOUNDS = {("below", 2): lambda r, a: np.percentile(r, 100 * (1 - a / 2)),      # upper end of the two-sided CI
          ("above", 2): lambda r, a: np.percentile(r, 100 * a / 2),            # lower end of the two-sided CI
          ("below", 1): lambda r, a: np.percentile(r, 100 * (1 - a)),          # one-sided upper bound (H2, H4b)
          ("above", 1): lambda r, a: np.percentile(r, 100 * a)}


def test_p_bound_inverts_every_registered_bound_exactly():
    """p < alpha exactly when the bound at level 1 - alpha lies beyond the null value, for each direction and for
    two-sided CIs and one-sided bounds. (Planted: a tail fraction, or a two-sided p for a one-sided bound, fails.)"""
    rng = np.random.default_rng(11)
    checked = 0
    for trial in range(300):
        n = int(rng.integers(3, 40))
        null = (0.0, 1.0, -1.0, 0.5)[trial % 4]
        reps = rng.normal(null + rng.normal(0, 1.5), 1.0, n)
        if trial % 9 == 0:
            reps[: n // 3] = null                                         # ties at the null value
        for (direction, sided), bound in BOUNDS.items():
            p = L.p_bound(reps, null, direction, sided)
            assert 0.0 <= p <= 1.0
            for a in rng.uniform(0.0005, 0.9995, 10):
                b = bound(reps, a)
                assert (p < a) == (b < null if direction == "below" else b > null), (trial, direction, sided, a, p)
                checked += 1
    assert checked == 12_000
    r = rng.normal(-1, 1, 50)
    assert L.p_bound(r, 0.0, "below", 2) == L.p_ci_inversion(r)          # H1's p is the special case
    assert L.p_bound([1.0, 2.0], 0.0, "below", 1) == 1.0 and L.p_bound([1.0, 2.0], 0.0, "above", 1) == 0.0
    assert math.isnan(L.p_bound([], 0.0, "below", 1))


def test_two_sided_p_is_the_smaller_directional_inversion():
    """H6a ("differs from 1"): p < alpha exactly when the two-sided CI excludes the null on either side."""
    rng = np.random.default_rng(5)
    for trial in range(200):
        reps = rng.normal(1.0 + rng.normal(0, 1.0), 0.5, int(rng.integers(5, 60)))
        p = F.p_differs(reps, 1.0)
        assert p == min(L.p_bound(reps, 1.0, "below", 2), L.p_bound(reps, 1.0, "above", 2))
        for a in rng.uniform(0.001, 0.999, 10):
            lo, hi = np.percentile(reps, [100 * a / 2, 100 * (1 - a / 2)])
            assert (p < a) == (lo > 1.0 or hi < 1.0)


def test_upper_95_bounds_are_inverted_one_sided_at_their_own_level():
    """H2 ("upper 95% bound ... at most Delta") and H4b ("upper 95% bound below 1") use the one-sided 95% bound (the
    95th percentile, as invariance computes upper95): with the null just above that bound p is just below 0.05, just
    below it p is above 0.05. (Planted: a two-sided p would be about 0.10 here.)"""
    from test_invariance import _sim
    from test_law_contrasts import _sim as _law
    from reach.analysis.s2.invariance import invariance_test
    recs = _sim(per_node=[0.6, 1.0, 1.4, 2.0], intercepts=[0, 0, 0, 0])
    up = invariance_test(recs, ["bits"], delta=1.0, boot=60, seed=3)["bits"]["upper95"]
    above = invariance_test(recs, ["bits"], delta=up + 1e-9, boot=60, seed=3)["bits"]
    below = invariance_test(recs, ["bits"], delta=up - 1e-9, boot=60, seed=3)["bits"]
    assert 0.03 < above["p_below_delta"] < 0.05 < below["p_below_delta"] < 0.07
    base = [dict(r, units=dict(bits_sem=float(r["K"]))) for r in _law(sd_target=0.5)]
    u = L.rate_ci(base, "bits_sem", boot=80)["upper95"]
    for c, ok in ((1 / u * (1 - 1e-6), True), (1 / u * (1 + 1e-6), False)):    # rescale: the rate scales by c
        rc = L.rate_ci([dict(r, units=dict(bits_sem=r["K"] * c)) for r in base], "bits_sem", boot=80)
        assert (rc["upper95"] < 1) == ok and (rc["p_below1"] < 0.05) == ok and 0.03 < rc["p_below1"] < 0.07
        assert rc["rate"] == pytest.approx(L.fit_law([dict(r, units=dict(bits_sem=r["K"] * c)) for r in base],
                                                     "bits_sem")["rate"])


# ------------------------------------------------------------------ Holm over a family with untestable members


def _m(i, p, kind="reject", untestable=None):
    return dict(id=i, p=p, kind=kind, untestable=untestable)


def test_holm_family_known_answer_counting_untestable_members():
    """m = 6 with two untestable members: p 0.001, 0.02, 0.30 (a 'covers 0' member), 0.012 -> adjusted 0.006, 0.08,
    0.9, 0.06. (Planted: m from testable members only (4) would support 0.012; a 'retain' member read as 'reject'
    would not be supported at 0.9.)"""
    mem = [_m("a", 0.001), _m("b", 0.02), _m("c", None, untestable="not built"), _m("d", 0.30, kind="retain"),
           _m("e", 0.012), _m("f", None, untestable="missing: no data")]
    s = F.holm_family(mem, alpha=0.05)
    assert s["m"] == 6 and s["alpha"] == 0.05
    assert [x["p_holm"] for x in mem] == [pytest.approx(0.006), pytest.approx(0.08), None, pytest.approx(0.9),
                                          pytest.approx(0.06), None]
    assert [x["supported"] for x in mem] == [True, False, False, True, False, False]
    assert s["supported"] == 2 and s["not_supported"] == 2 and s["untestable"] == 2
    assert s["untestable_reasons"] == {"not built": 1, "missing: no data": 1}
    assert mem[2]["status"].startswith("untestable") and "not supported" in mem[2]["status"]
    # 'retain' (the claim is that the CI covers 0): a small adjusted p is evidence against it; exactly alpha covers 0
    r = [_m("a", 0.001, "retain"), _m("b", 0.05, "retain")]
    F.holm_family(r, alpha=0.05)
    assert [x["p_holm"] for x in r] == [pytest.approx(0.002), pytest.approx(0.05)]
    assert [x["supported"] for x in r] == [False, True]
    # 'reject' is strict: an adjusted p of exactly alpha does not exclude the null
    e = [_m("a", 0.025), _m("b", 0.025)]
    F.holm_family(e, alpha=0.05)
    assert not any(x["supported"] for x in e)
    # a NaN p without a stated reason is untestable (not estimable), never a crash or a support
    n = [_m("a", float("nan")), _m("b", 0.0)]
    s = F.holm_family(n, alpha=0.05)
    assert n[0]["p_holm"] is None and not n[0]["supported"] and n[1]["supported"] and s["m"] == 2
    with pytest.raises(ValueError, match="twice"):
        F.holm_family([_m("a", 0.1), _m("a", 0.2)], alpha=0.05)


def test_h8_member_is_the_conjunction_of_its_parts():
    """H8 (cfg x set): "H1 holds, the sign of GP minus null is unchanged, and the set x configuration interaction is
    within Delta" -> the member's p is the largest part's (intersection-union); a part without a p makes the member
    untestable. (Planted: the smallest part's p would pass a member whose weakest part fails.)"""
    assert F.conjunction({"H8a": (0.01, None), "H8b": (0.002, None), "H8c": (0.03, None)}) == (0.03, None)
    p, why = F.conjunction({"H8a": (0.01, None), "H8b": (None, "not built"), "H8c": (None, "not built")})
    assert p is None and why == "H8b not built, H8c not built"
    cells = [dict(set="S", config="R1", budget=1, p_below0=0.001, estimable=True, saturated=False),
             dict(set="S", config="R1", budget=4, p_below0=0.02, estimable=True, saturated=False)]
    assert F.h8a_part(cells, ("S", "R1")) == (0.02, None)                  # H1 holds at both budgets
    assert F.h8a_part(cells[:1], ("S", "R1")) == (None, "missing at budget 4")
    sat = [dict(cells[0], saturated=True, estimable=False, p_below0=float("nan")), cells[1]]
    assert F.h8a_part(sat, ("S", "R1")) == (None, "saturated at budget 1")


def test_resolution_of_the_smallest_holm_threshold_at_the_registered_b():
    """With B = 2,000, a percentile bound at level 1 - alpha/m has B * alpha / (m * sides) replicates beyond it.
    Fewer than one means the decision rests on the one or two most extreme replicates (interpolated)."""
    r = F.resolution(61, sided=2, boot=2000, alpha=0.05)
    assert r["threshold"] == pytest.approx(0.05 / 61) and r["replicates_beyond"] == pytest.approx(0.8197, abs=1e-4)
    assert not r["resolvable"] and not r["conventional_reaches"]           # 2/2001 > 0.05/61
    r = F.resolution(4, sided=1, boot=2000, alpha=0.05)
    assert r["replicates_beyond"] == pytest.approx(25.0) and r["resolvable"] and r["conventional_reaches"]
    r = F.resolution(18, sided=2, boot=2000, alpha=0.05)
    assert r["replicates_beyond"] == pytest.approx(2.7778, abs=1e-4) and r["resolvable"]
    assert F.resolution(6, sided=None, boot=2000, alpha=0.05)["resolvable"]    # analytic p (H4a): no bootstrap limit


# ------------------------------------------------------------------ the declared families


def test_declared_families_have_the_filed_sizes():
    from reach.runner.phases import CAL_SETS, FULL_SETS
    fam = F.declared()
    size = {f: {d: len(v) for d, v in fam[f].items()} for f in fam}
    assert size == {"H2": {"real": 4, "bool": 4}, "H4": {"real": 6, "bool": 5}, "H5": {"real": 18, "bool": 18},
                    "H6": {"real": 3, "bool": 3}, "H8": {"real": 18, "bool": 18}, "H9a": {"real": 6}}
    assert fam["H2"]["real"] == [("nodes",), ("bits_uniform",), ("bits_syn",), ("bits_sem",)]
    assert [m[0] for m in fam["H4"]["real"]] == ["H4a", "H4b", "H4c", "H4d", "H4e", "H4f"]
    assert "H4b" not in [m[0] for m in fam["H4"]["bool"]]                   # "H4b: in the real domain"
    assert {m[0] for d in ("real", "bool") for m in fam["H5"][d]} == set(FULL_SETS)
    assert {m[1] for m in fam["H5"]["bool"]} == {"A-B", "HC-A", "B-RS"}
    assert {m[0] for m in fam["H9a"]["real"]} == {"R-ARITH4", "R-ELEM4", "R-TRIG"} <= set(CAL_SETS)
    assert {(m[1], m[3]) for m in fam["H9a"]["real"]} == {("operon", 16)}
    assert {m[1] for m in fam["H8"]["real"]} == {f"R{i}" for i in range(1, 7)}


# ------------------------------------------------------------------ wiring into confirm


def _operon_rows(g, cfg, beta, seed):
    rng = np.random.default_rng(seed)
    out = []
    for K in range(5, 25, 2):
        for t in range(8):
            for s in range(3):
                p = 1 / (1 + math.exp(-(2.0 + beta * (K - 12))))
                out.append(dict(key=f"{g}|{cfg}|{K}|{t}|{s}", set=g, target_id=f"{g}_K{K}_{t}", seed=s,
                                K_construction=K, K_proven=K, algorithm="operon", config=cfg, budget=16,
                                exact=bool(rng.random() < p), terminal_state="budget_exhausted", cpu_s=1.0,
                                units=dict(nodes=float(K), bits_uniform=2.0 * K, bits_syn=1.5 * K, bits_sem=float(K))))
    return out


def _calibration(tmp_path):
    from test_dryrun import SETS, _rows, _write
    _write(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)])
    _write(tmp_path, "s2_cal_mech", _rows("BOOL-NAND-n5", "B", 0.5, 9) + _rows("BOOL-NAND-n5", "HC", 2.0, 8))
    _write(tmp_path, "s2_cal_robust", [dict(r, config="R1", key=r["key"] + "|R1")
                                       for r in _rows("BOOL-NAND-n5", "A", 2.0, 5)])
    _write(tmp_path, "s2_cal_operon", _operon_rows("R-TRIG", "symbolic", -0.6, 1) + _operon_rows("R-TRIG", "default",
                                                                                                    0.0, 2))


def test_confirm_reports_every_declared_family(tmp_path):
    from reach.analysis.s2 import confirm as C
    _calibration(tmp_path)
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=30, allow_incomplete=True)
    h = res["holm"]
    assert {f: {d: h[f][d]["m"] for d in h[f]} for f in h} == {
        "H2": {"real": 4, "bool": 4}, "H4": {"real": 6, "bool": 5}, "H5": {"real": 18, "bool": 18},
        "H6": {"real": 3, "bool": 3}, "H8": {"real": 18, "bool": 18}, "H9a": {"real": 6}}
    assert all(h[f][d]["family"] == "declared" and h[f][d]["alpha"] == 0.05 for f in h for d in h[f])
    for f in h:                                                       # untestable members are never supported
        for d in h[f]:
            for mm in h[f][d]["members"]:
                assert mm["supported"] is False or mm["p_holm"] is not None
                assert h[f][d]["resolution"]["boot"] == 2000                  # judged at the registered B
    # H2: one-sided 95% bound against Delta (the existing 'upper95'); existing fields kept
    for u, v in res["h2"]["bool"]["units"].items():
        assert {"excess", "upper95", "invariant", "boot"} <= set(v)
        assert (v["p_below_delta"] < 0.05) == (v["upper95"] < 0.0059)
    assert all(mm["untestable"] == "missing: no data" for mm in h["H2"]["real"]["members"])
    # H5: A - B above 0; HC - A above -1 node (within +-1 or positive); B - RS built (D080 c), no random-sampling phase here
    ab, ha = res["h5"]["BOOL-NAND-n5"]["A-B"], res["h5"]["BOOL-NAND-n5"]["HC-A"]
    assert {"diff", "lo", "hi", "targets", "dropped", "estimable"} <= set(ab) and "reps" not in ab
    assert (ab["p"] < 0.05) == (ab["lo"] > 0) and (ha["p"] < 0.05) == (ha["lo"] > -1.0)
    mem = {tuple(mm["id"]): mm for mm in h["H5"]["bool"]["members"]}
    brs = mem[("BOOL-NAND-n5", "B-RS")]
    assert brs["untestable"] == "missing: no data" and not brs["supported"]
    assert mem[("BOOL-NOR-n5", "A-B")]["untestable"] == "missing: no data"
    assert mem[("BOOL-NAND-n5", "A-B")]["p"] == ab["p"]
    # H4b (real only) and H6a: the pooled bits_sem rate with a target bootstrap within sets
    rc = res["h4"]["bool"]["rate_bits_sem"]
    assert rc["rate"] == pytest.approx(res["h4"]["bool"]["law_bits_sem"]["rate"])
    assert (rc["p_below1"] < 0.05) == (rc["upper95"] < 1.0) and (rc["p_differs1"] < 0.05) == (rc["lo"] > 1 or rc["hi"] < 1)
    h6 = {mm["id"][0]: mm for mm in h["H6"]["bool"]["members"]}
    assert h6["H6a"]["p"] == rc["p_differs1"]
    # H6b and H6c built (D081 b, c; row 109): their p is the result's; H6c needs both domains (no real data here)
    assert h6["H6b"]["p"] == res["h6b"]["bool"]["p"] and h6["H6b"]["untestable"] == res["h6b"]["bool"]["untestable"]
    assert h6["H6c"]["untestable"] == "missing: no data in real" and not h6["H6c"]["supported"]
    h4 = {mm["id"][0]: mm for mm in h["H4"]["bool"]["members"]}
    assert h4["H4a"]["kind"] == "retain" and h4["H4a"]["p"] == res["h4"]["bool"]["curvature_nodes"]["p_wald"]
    # H8: conjunction members; H8a from the H1-type cells; H8b and H8c built (D081 d, e; row 109)
    m8 = {tuple(mm["id"]): mm for mm in h["H8"]["bool"]["members"]}
    nand = m8[("BOOL-NAND-n5", "R1")]                  # R1 on one set only: H8c has no other set to transfer from
    assert nand["p_h8b"] == res["h8b"]["BOOL-NAND-n5|R1"]["p"] is not None
    assert nand["untestable"] == "H8c fewer than 2 robustness sets with this configuration"
    assert m8[("BOOL-NOR-n5", "R1")]["untestable"].startswith("H8a missing")
    assert h["H8"]["bool"]["supported"] == 0
    # H9a: H1-type cells on Operon rows at 16 x E0, one Holm family (real)
    cells = {(c["set"], c["config"]): c for c in res["h9a"]}
    assert set(cells) == {("R-TRIG", "symbolic"), ("R-TRIG", "default")}
    assert cells[("R-TRIG", "symbolic")]["holm_supported"] and not cells[("R-TRIG", "default")]["holm_supported"]
    assert len(h["H9a"]["real"]["missing"]) == 4
    # report: a summary per family and domain; members in their own file
    text = C.render(res, "t", "s", members_file="m.md")
    assert "## Holm families" in text and all(f"**{f}, " in text for f in h)
    assert len(text.splitlines()) <= 200
    table = C.render_holm_members(res, "t", "s")
    assert table.count("\n| ") == sum(h[f][d]["m"] for f in h for d in h[f]) + len(h)     # H8a cells: render_h8
    parts = C.render_h8(res, "t", "s")
    assert parts.count("\n| ") == len(res["h8"]) + len(res["h8b"]) + 2
    assert not any("Holm for the" in x for x in res["not_run"])
    json.dumps(res, default=float)


def test_dryrun_families_are_the_observed_members(tmp_path):
    from reach.analysis.s2 import confirm as C
    _calibration(tmp_path)
    pilot = dict(C.PHASE_MAPS["pilot"], reach=("s2_cal_a16",), mech=("s2_cal_mech",), robust=("s2_cal_robust",),
                 operon=("s2_cal_operon",))
    res = C.build(pilot, tmp_path, boot=20)
    h = res["holm"]
    assert h["H5"]["bool"]["family"] == "observed" and h["H5"]["bool"]["m"] == 3      # NAND x 3 contrasts
    assert h["H8"]["bool"]["m"] == 1 and h["H9a"]["real"]["m"] == 2 and "real" not in h["H2"]
    assert h["H4"]["bool"]["m"] == 5 and h["H6"]["bool"]["m"] == 3                   # not set-indexed: all members

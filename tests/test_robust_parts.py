"""H8b and H8c as filed in Stage 1, and the step-C members wired into confirm (REQ-ANA-52 to 54; D081 d-e; guardrail
row 109), written before the code. Synthetic data only.

Stage 1 (OSF a3spg), as filed: "H8: under each robustness configuration R1-R6, H1 holds, the sign of GP minus null is
unchanged, and the set x configuration interaction is within Delta." Design at 3b8aa5e (hypotheses.md, H8): "H8b: the
sign of GP - null per domain equals the base configuration's." "H8c (key): the set x configuration interaction in the
H2-preferred unit stays within Delta_cfg: a configuration shifts reach about equally in every set."
"""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import cfgint as CI
from reach.analysis.s2 import law as L
from reach.analysis.s2 import nullgap as N
from reach.analysis.s2.load import E0

LEVELS = tuple(range(6, 26, 2))


def _curve(g, config, k50, b1=1.5, seed=0, targets=20, runs=5, unit_scale=1.0):
    """Records at budgets 1 and 4 (k50: {budget: K50 in bits_sem}); bits_sem = unit_scale x K; same target ids and
    seeds in every configuration (paired)."""
    rng = np.random.default_rng(seed)
    out = []
    for K in LEVELS:
        for t in range(targets):
            for s in range(runs):
                for b, k in k50.items():
                    p = 1 / (1 + math.exp(b1 * (unit_scale * K - k)))
                    out.append(dict(set=g, target_id=f"{g}_K{K}_{t}", seed=s, K=K, K_proven=K, budget=b, config=config,
                                    algorithm="A", y=int(rng.random() < p),
                                    units=dict(nodes=float(K), bits_uniform=2.0 * K, bits_syn=1.5 * K,
                                               bits_sem=unit_scale * K)))
    return out


NULL = {1: math.log2(E0), 4: math.log2(4 * E0)}


def test_h8b_same_sign_as_the_base_at_e0_and_4x_known_answers():
    base = _curve("A", "base", {1: 10.0, 4: 12.0}, seed=1)                   # GP below the null at both budgets
    same = _curve("A", "R1", {1: 12.0, 4: 13.0}, seed=2)
    h = N.h8b(base, same, boot=100)
    assert set(h["budgets"]) == {1, 4} and h["untestable"] is None
    for b in (1, 4):
        v = h["budgets"][b]
        assert v["base_gap"] == pytest.approx({1: 10.0, 4: 12.0}[b] - NULL[b], abs=0.3)
        assert v["config_gap"] == pytest.approx({1: 12.0, 4: 13.0}[b] - NULL[b], abs=0.3)
    assert h["p"] == max(h["budgets"][b]["p"] for b in (1, 4)) and h["p"] < 0.05
    flip = _curve("A", "R2", {1: 18.0, 4: 20.0}, seed=3)                       # GP above the null: the sign changes
    assert N.h8b(base, flip, boot=100)["p"] == 1.0
    late = _curve("A", "R3", {1: 12.0, 4: 19.0}, seed=4)                       # unchanged at E0, changed at 4 x E0
    h = N.h8b(base, late, boot=100)
    assert h["budgets"][1]["p"] < 0.05 and h["p"] == h["budgets"][4]["p"] > 0.05
    level = _curve("A", "base", {1: NULL[1], 4: NULL[4]}, seed=5)              # base gap about 0: its sign is unknown
    h = N.h8b(level, same, boot=100)
    assert h["p"] > 0.05                                                        # not "unchanged" by the base's point sign


def test_h8b_is_paired_by_target_and_untestable_without_data():
    base = _curve("A", "base", {1: 10.0, 4: 12.0}, seed=1)
    cfg = _curve("A", "R1", {1: 12.0, 4: 13.0}, seed=2)
    h = N.h8b(base, cfg, boot=50)
    assert h["targets"] == len(LEVELS) * 20
    h = N.h8b(base, [r for r in cfg if r["budget"] == 1], boot=20)
    assert h["p"] is None and h["untestable"] == "missing at budget 4"


def _three(shifts, slopes=(0.5, 1.0, 2.0), k0=(10.0, 12.0, 14.0), seed=0, targets=40):
    base, cfg = {}, {}
    for i, (g, b1, k, d) in enumerate(zip(("A", "B", "C"), slopes, k0, shifts)):
        base[g] = _curve(g, "base", {1: k}, b1=b1, seed=seed + 10 * i, targets=targets)
        cfg[g] = _curve(g, "R1", {1: k + d}, b1=b1, seed=seed + 10 * i + 5, targets=targets)
    return base, cfg


DELTA = 0.02          # a test margin; confirm passes the registered Delta of the domain


def test_h8c_equal_reach_shifts_are_within_delta_and_a_planted_interaction_is_not():
    """A configuration that shifts reach by 2 bits in every set, on sets with slopes 0.5, 1 and 2 per bit: the excess
    log-loss of the other sets' shift over the set's own (5-fold over targets) is about 0, within +-Delta (two-sided
    95% CI). A logit-scale shift (1, 2 and 4) would not transfer: the shift is in the unit's reach. Set C shifted by
    6 bits instead: C's interaction exceeds Delta."""
    base, cfg = _three((2.0, 2.0, 2.0))
    h = CI.h8c(base, cfg, "bits_sem", delta=DELTA, boot=60)
    assert set(h) == {"A", "B", "C"}
    for g, v in h.items():
        assert abs(v["excess"]) < 0.005 and v["shift_own"] == pytest.approx(2.0, abs=0.5)
        assert (v["p"] < 0.05) == (-DELTA < v["lo"] and v["hi"] < DELTA) and v["estimable"]
        assert v["holds"] == (v["p"] < 0.05) and v["delta"] == DELTA
    assert all(v["holds"] for v in h.values())
    for v in h.values():                     # equivalence: the two-sided 95% CI, both ends (not H2's one-sided bound)
        assert v["p"] == max(L.p_bound(v["_reps"], -DELTA, "above", 2), L.p_bound(v["_reps"], DELTA, "below", 2))
    assert any(0 < v["p"] < 1 for v in h.values())
    base, cfg = _three((2.0, 2.0, 6.0))
    h = CI.h8c(base, cfg, "bits_sem", delta=DELTA, boot=60)
    assert h["C"]["excess"] > DELTA and not h["C"]["holds"] and h["C"]["p"] > 0.05
    assert h["C"]["shift_logo"] == pytest.approx(2.0, abs=0.5) and h["C"]["shift_own"] == pytest.approx(6.0, abs=0.8)
    assert all(v["p"] == 1.0 for v in h.values())          # every replicate beyond the margin: the interaction leaks


def test_h8_member_is_the_conjunction_of_the_built_parts():
    """An H8 member holds when H8a, H8b and H8c all hold: p = the largest of the three; a part without a p makes the
    member untestable, with the part named."""
    from reach.analysis.s2 import families as F
    cells = [dict(set="S", config="R1", budget=b, p_below0=p, estimable=True, saturated=False)
             for b, p in ((1, 0.001), (4, 0.002))]
    res = dict(h8=cells, h8b={"S|R1": dict(p=0.003, untestable=None)}, h8c={"S|R1": dict(p=0.0005, untestable=None)})
    mm = F._h8(res, "bool", ("S", "R1"))
    assert mm["p"] == 0.003 and mm["untestable"] is None and (mm["p_h8a"], mm["p_h8b"], mm["p_h8c"]) == (0.002, 0.003,
                                                                                                         0.0005)
    res["h8c"]["S|R1"]["p"] = 0.04
    assert F._h8(res, "bool", ("S", "R1"))["p"] == 0.04
    res["h8b"]["S|R1"] = dict(p=None, untestable="not estimable at budget 4")
    mm = F._h8(res, "bool", ("S", "R1"))
    assert mm["p"] is None and mm["untestable"] == "H8b not estimable at budget 4"
    del res["h8b"]["S|R1"]
    assert F._h8(res, "bool", ("S", "R1"))["untestable"] == "H8b missing"


def test_h8c_reads_the_unit_and_needs_two_sets():
    base, cfg = _three((2.0, 2.0, 2.0))
    one = CI.h8c({"A": base["A"]}, {"A": cfg["A"]}, "bits_sem", DELTA, boot=10)
    assert one["A"]["p"] is None and one["A"]["untestable"] == "fewer than 2 robustness sets with this configuration"
    # in nodes the same data are the same shift only when bits_sem = nodes; with bits_sem = 2 x nodes in set C,
    # a 2-node shift is a 4-bit shift there, so the shift in bits_sem is unequal and bits_sem shows an interaction
    base["C"] = _curve("C", "base", {1: 28.0}, b1=1.0, seed=40, targets=40, unit_scale=2.0)
    cfg["C"] = _curve("C", "R1", {1: 32.0}, b1=1.0, seed=45, targets=40, unit_scale=2.0)
    for g in ("A", "B"):
        base[g] = [dict(r, units=dict(r["units"], nodes=r["units"]["bits_sem"])) for r in base[g]]
        cfg[g] = [dict(r, units=dict(r["units"], nodes=r["units"]["bits_sem"])) for r in cfg[g]]
    hn = CI.h8c(base, cfg, "nodes", DELTA, boot=40)
    hb = CI.h8c(base, cfg, "bits_sem", DELTA, boot=40)
    assert hn["C"]["shift_own"] == pytest.approx(2.0, abs=0.5) and hb["C"]["shift_own"] == pytest.approx(4.0, abs=1.0)
    assert hn["C"]["excess"] < hb["C"]["excess"]


# ------------------------------------------------------------------ wiring into confirm (declared and observed)


def _robust_root(tmp_path, censored=False):
    from test_dryrun import SETS, _rows, _write
    a16 = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + 0.5 * i, i)]
    if censored:
        a16.append(dict(a16[0], key=a16[0]["key"] + "|t", seed=99, gen_exact=None, gens_reached=50,
                        terminal_state="timeout"))
    _write(tmp_path, "s2_cal_a16", a16)
    rob = []
    for j, g in enumerate(("BOOL-NAND-n5", "BOOL-NOR-n5")):
        rob += [dict(r, config="R1", key=r["key"] + "|R1") for r in _rows(g, "A", 2.5 + 0.5 * j, 30 + j)]
    _write(tmp_path, "s2_cal_robust", rob)
    return tmp_path


def test_confirm_wires_h4c_h6b_h6c_h8b_h8c_into_their_holm_families(tmp_path, monkeypatch):
    from reach.analysis.s2 import confirm as C
    root = _robust_root(tmp_path, censored=True)
    seen = []
    real_h8b = N.h8b
    monkeypatch.setattr(N, "h8b", lambda base, cfg, **k: seen.append((base, cfg)) or real_h8b(base, cfg, **k))
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), root, boot=20, allow_incomplete=True)
    h = res["holm"]
    assert {f: {d: h[f][d]["m"] for d in h[f]} for f in ("H4", "H6", "H8")} == {
        "H4": {"real": 6, "bool": 5}, "H6": {"real": 3, "bool": 3}, "H8": {"real": 18, "bool": 18}}
    h4 = {mm["id"][0]: mm for mm in h["H4"]["bool"]["members"]}
    c = res["h4"]["bool"]["h4c"]
    assert h4["H4c"]["p"] == c["p"] and h4["H4c"]["untestable"] == c["untestable"] and h4["H4c"]["kind"] == "reject"
    assert c["margin"] == 1.0 and len(c["pairs"]) == 3 and c["unit"] == "bits_sem"
    h6 = {mm["id"][0]: mm for mm in h["H6"]["bool"]["members"]}
    assert h6["H6b"]["p"] == res["h6b"]["bool"]["p"] and res["h6b"]["bool"]["unit"] == "nodes"
    assert h6["H6c"]["untestable"] == "missing: no data in real" == res["h6c"]["untestable"]
    assert {mm["id"][0]: mm["untestable"] for mm in h["H6"]["real"]["members"]}["H6c"] == "missing: no data in real"
    # H8: base runs paired with the configuration's targets and seeds, at E0 and 4 x E0
    for base, cfg in seen:
        assert {(r["target_id"], r["seed"]) for r in base} <= {(r["target_id"], r["seed"]) for r in cfg}
        assert {r["budget"] for r in base} == {1, 4} and {r["config"] for r in base} == {"base"}
    m8 = {tuple(mm["id"]): mm for mm in h["H8"]["bool"]["members"]}
    nand = m8[("BOOL-NAND-n5", "R1")]
    b, ci = res["h8b"]["BOOL-NAND-n5|R1"], res["h8c"]["BOOL-NAND-n5|R1"]
    assert ci["unit"] == res["h2"]["bool"]["report_unit"] and ci["delta"] == 0.0059
    parts = [nand["p_h8a"], b["p"], ci["p"]]
    if all(p is not None for p in parts):
        assert nand["p"] == max(parts)
    else:
        assert nand["p"] is None and "not built" not in nand["untestable"]
    assert m8[("BOOL-RICH-n5", "R1")]["untestable"].startswith("H8a missing")
    assert not any(k in x for x in res["not_run"] for k in ("H4c", "H6b", "H6c", "H8b", "H8c"))
    s = res["sensitivity_cpu_excluded"]
    assert s["recomputed"] and {"h6b", "h6c", "h8b", "h8c"} <= set(C.SENSITIVITY_KEYS) <= set(s)
    assert "BOOL-NAND-n5|R1" in s["h8c"] and "p" in s["h4"]["bool"]["h4c"]
    js = json.dumps(res, default=float)
    assert "_reps" not in js


def test_dryrun_reports_the_observed_h8_parts(tmp_path):
    from reach.analysis.s2 import confirm as C
    root = _robust_root(tmp_path)
    pilot = dict(C.PHASE_MAPS["pilot"], reach=("s2_cal_a16",), robust=("s2_cal_robust",), mech=())
    res = C.build(pilot, root, boot=20)
    h = res["holm"]
    assert h["H8"]["bool"]["family"] == "observed" and h["H8"]["bool"]["m"] == 2
    assert set(res["h8b"]) == set(res["h8c"]) == {"BOOL-NAND-n5|R1", "BOOL-NOR-n5|R1"}
    text = C.render_h8(res, "t", "s")
    assert text.count("\n| BOOL-") == len(res["h8"]) + 2 and len(text.splitlines()) <= 200
    assert "H8b" in C.render(res, "t", "s") and "H4c" in C.render_slopes_gaps(res, "t", "s")


def test_h8_report_states_that_h8c_is_underpowered():
    """D082 (ii), the author's decision on D081 (e): no deviation; the H8 report says openly that H8c, and so most H8
    members, are underpowered at the registered size, so a "not supported" H8 member is not evidence of an
    interaction."""
    from reach.analysis.s2 import confirm as C
    text = C.render_h8(dict(label="t", h8=[], h8b={}, h8c={}), "t", "s")
    line = next(x for x in text.splitlines() if "underpowered" in x)
    assert "H8c" in line and "registered size" in line and "D081 (e)" in line and "D082" in line
    assert "not evidence of an interaction" in line


def test_h6b_and_h6c_read_e0_records_only(tmp_path, monkeypatch):
    """Guardrail row 114 (independent review, 2026-10-27): confirm's filter of H6b's and H6c's records to E0 (D081 b,
    c: "at E0") was untested; passing the records at every budget went unnoticed. Spies on nullgap.h6b and
    nullgap.domain_gap: every record passed is at budget 1, with every target of each set (11 levels x 8 targets) and
    its 3 seeds."""
    from reach.analysis.s2 import confirm as C
    root = _robust_root(tmp_path)
    seen = []
    real_h6b, real_gap = N.h6b, N.domain_gap
    monkeypatch.setattr(N, "h6b", lambda recs, *a, **k: seen.append(("h6b", recs)) or real_h6b(recs, *a, **k))
    monkeypatch.setattr(N, "domain_gap", lambda recs, *a, **k: seen.append(("gap", recs)) or real_gap(recs, *a, **k))
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), root, boot=10, allow_incomplete=True)
    assert sorted(k for k, _ in seen) == ["gap", "h6b"] and "bool" in res["h6b"]
    for _, by_set in seen:
        assert sorted(by_set) == sorted(SETS_A16)
        for g, recs in by_set.items():
            assert {r["budget"] for r in recs} == {1}
            assert len({r["target_id"] for r in recs}) == 88 and len(recs) == 88 * 3


SETS_A16 = ("BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5")

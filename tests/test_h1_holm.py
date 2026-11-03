"""H1's registered Holm step and H10a's (REQ-ANA-28; guardrail row 95), written before the code. Synthetic data only.

Stage 1 (OSF a3spg), Inference criteria: "Directional effects: the 95% CI excludes 0 in the predicted direction. ...
Holm correction within families declared now, each counting untestable members: per domain, H1 cells; ... P-values for
Holm come from the bootstrap by CI inversion." Saturated cells: "counted as an untestable family member, not as a
failure"."""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import contrasts
from reach.analysis.s2 import law as L


# ------------------------------------------------------------------ p by inversion of the percentile CI


def test_ci_inversion_p_agrees_with_the_percentile_ci_at_every_level():
    """p < alpha exactly when the two-sided 100(1 - alpha)% percentile CI (np.percentile, as slope_ci) lies below 0.
    Small replicate sets make the interpolation matter, so a plain tail fraction would disagree."""
    rng = np.random.default_rng(7)
    checked = 0
    for trial in range(400):
        n = int(rng.integers(3, 40))
        reps = rng.normal(rng.normal(-1.0, 1.5), 1.0, n)
        if trial % 7 == 0:
            reps[: n // 3] = 0.0                                   # ties at exactly 0
        p = L.p_ci_inversion(reps)
        assert 0.0 <= p <= 1.0
        for a in rng.uniform(0.0005, 0.9995, 25):
            hi = np.percentile(reps, 100 * (1 - a / 2))
            lo = np.percentile(reps, 100 * a / 2)
            assert (p < a) == (hi < 0 and lo < 0), (trial, a, p, hi)
            checked += 1
    assert checked == 10_000
    assert L.p_ci_inversion([-3.0, -2.0, -1.0]) == 0.0              # every CI excludes 0 below
    assert L.p_ci_inversion([0.0, 1.0, 2.0]) == 1.0                 # no CI lies below 0
    assert L.p_ci_inversion([1.0, 2.0, 3.0]) == 1.0                 # wrong direction: not evidence (directional)
    assert math.isnan(L.p_ci_inversion([]))


def _cell(beta, seed, n_targets=15, levels=range(4, 24, 2), seeds=5, alpha=6.0):
    rng = np.random.default_rng(seed)
    recs = []
    for K in levels:
        for t in range(n_targets):
            for s in range(seeds):
                eta = alpha + beta * (K - 13)
                recs.append(dict(set="S", target_id=f"K{K}_{t}", seed=s, K=K, budget=1, units=dict(nodes=float(K)),
                                 y=int(rng.random() < 1 / (1 + math.exp(-eta)))))
    return recs


def test_slope_ci_p_is_directional_and_matches_its_95_ci():
    neg = L.slope_ci(_cell(-0.5, 1, alpha=0.0), boot=300)
    assert neg["hi"] < 0 and neg["p_below0"] < 0.05 and not neg["saturated"]
    pos = L.slope_ci(_cell(+0.5, 2, alpha=0.0), boot=300)            # planted: recovery rises with size
    assert pos["lo"] > 0 and pos["p_below0"] == 1.0                  # a two-sided p would be small here
    for k in range(10):                                              # null cells: decision equals the CI's
        rng = np.random.default_rng(50 + k)
        c = L.slope_ci([dict(r, y=int(rng.random() < 0.5)) for r in _cell(0.0, 3)], boot=200, seed=k)
        assert (c["p_below0"] < 0.05) == (c["hi"] < 0)
    sat = L.slope_ci([dict(r, y=1) for r in _cell(-0.5, 4)], boot=50)
    assert sat["saturated"] and not sat["estimable"]


# ------------------------------------------------------------------ Holm over a declared family, per domain

DOM = {"R1": "real", "R2": "real", "R3": "real", "R4": "real", "B1": "bool", "B2": "bool"}.get


def _c(g, p, estimable=True, saturated=False, budget=1, algorithm="A"):
    return dict(set=g, algorithm=algorithm, config="base", budget=budget, p_below0=p, estimable=estimable,
                saturated=saturated)


def test_holm_known_answer_per_domain_counting_untestable_members():
    """Real family of 4: p 0.001, 0.02, saturated, 0.03 -> adjusted 0.004, 0.06, None, 0.06: only the first holds.
    (Planted errors: m from testable cells only would support all three; pooling the domains (m = 6) would fail B1 and
    B2; no adjustment would support 0.02 and 0.03.) Boolean family of 2: 0.02, 0.024 -> 0.04, 0.04: both hold. A
    not-estimable cell (more than 10% degenerate replicates) is in m and not supported."""
    cells = [_c("R1", 0.001), _c("R2", 0.02), _c("R3", float("nan"), estimable=False, saturated=True), _c("R4", 0.03),
             _c("B1", 0.02), _c("B2", 0.024)]
    out = L.holm_cells(cells, DOM, alpha=0.05)
    r, b = out["real"], out["bool"]
    assert r["m"] == 4 and b["m"] == 2
    assert [c["p_holm"] for c in cells[:4]] == [pytest.approx(0.004), pytest.approx(0.06), None, pytest.approx(0.06)]
    assert [c["holm_supported"] for c in cells] == [True, False, False, False, True, True]
    assert cells[2]["holm_status"] == "saturated (untestable)" and cells[1]["holm_status"] == "not supported"
    assert r["supported"] == 1 and r["not_supported"] == 2 and r["saturated"] == 1 and not r["all_testable_supported"]
    assert b["supported"] == 2 and b["all_testable_supported"] and b["alpha"] == 0.05
    # strict: an adjusted p of exactly alpha is not below it (the CI's bound at 0 does not exclude 0)
    edge = [_c("B1", 0.025), _c("B2", 0.025)]
    L.holm_cells(edge, DOM, alpha=0.05)
    assert edge[0]["p_holm"] == 0.05 and not edge[0]["holm_supported"]
    # not estimable but not saturated: an untestable member that counts as not supported
    ne = [_c("B1", 0.0, estimable=False), _c("B2", 0.0)]
    s = L.holm_cells(ne, DOM, alpha=0.05)["bool"]
    assert ne[0]["p_holm"] is None and ne[0]["holm_status"] == "not estimable (not supported)"
    assert ne[1]["p_holm"] == 0.0 and s["m"] == 2 and s["not_estimable"] == 1 and not s["all_testable_supported"]
    # every all-saturated testable set: saturated cells alone do not make the domain fail
    ok = [_c("B1", 0.0), _c("B2", float("nan"), estimable=False, saturated=True)]
    assert L.holm_cells(ok, DOM, alpha=0.05)["bool"]["all_testable_supported"]


def test_declared_family_counts_missing_cells_and_refuses_undeclared_ones():
    declared = [("B1", "A", "base", 1), ("B2", "A", "base", 1), ("B2", "A", "base", 16)]
    cells = [_c("B1", 0.01), _c("B2", 0.01)]
    s = L.holm_cells(cells, DOM, alpha=0.05, declared=declared)["bool"]
    assert s["m"] == 3 and s["missing"] == [["B2", "A", "base", 16]] and s["family"] == "declared"
    assert [c["p_holm"] for c in cells] == [pytest.approx(0.03), pytest.approx(0.03)]       # m = 3, not 2
    assert not s["all_testable_supported"]                                                  # a missing cell holds nothing
    with pytest.raises(ValueError, match="not in the declared"):
        L.holm_cells([_c("B1", 0.01, budget=4)], DOM, alpha=0.05, declared=declared)
    obs = L.holm_cells([_c("B1", 0.01), _c("B2", 0.01)], DOM, alpha=0.05)["bool"]
    assert obs["family"] == "observed" and obs["m"] == 2 and obs["missing"] == []


# ------------------------------------------------------------------ wiring: confirm (H1) and cgp_confirm (H10a)


def _write(root, phase, rows):
    d = root / "data/runs" / phase
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(n_tasks=len(rows), tasks=[])))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _calibration(tmp_path):
    from test_dryrun import SETS, _rows
    _write(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)])
    _write(tmp_path, "s2_cal_mech", _rows("BOOL-NAND-n5", "B", 0.5, 9))


def test_confirm_reports_holm_over_the_declared_h1_family(tmp_path):
    from reach.analysis.s2 import confirm as C
    from reach.runner.phases import CAL_SETS, FULL_SETS
    assert C.REGISTERED["alpha"] == 0.05
    fam = C.declared_h1_cells()
    assert len(fam) == len(CAL_SETS) * len(C.BUDGETS) + 2 * len(FULL_SETS) == 136
    _calibration(tmp_path)
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=30, allow_incomplete=True)
    h = res["h1_holm"]
    assert h["real"]["m"] == 61 and h["bool"]["m"] == 75 and h["real"]["family"] == "declared"
    assert h["bool"]["m"] == len(res["h1"]) + len(h["bool"]["missing"])          # every present cell is Boolean here
    for c in res["h1"]:
        assert {"slope", "lo", "hi", "dropped", "estimable", "targets", "n"} <= set(c)      # existing output kept
        assert "p_holm" in c and "holm_supported" in c and "holm_status" in c
        assert not c["holm_supported"] or c["hi"] < 0                                     # Holm never adds support
    assert not any("Holm: step 3b" in x for x in res["not_run"])
    text = C.render(res, "t", "s")
    assert "Holm" in text and "61" in text and "75" in text
    table = C.render_h1_cells(res, "t", "s")                       # per-cell table in its own file (200-line limit)
    assert table.count("\n| ") == len(res["h1"]) + 1
    pilot = C.build(dict(C.PHASE_MAPS["pilot"], reach=("s2_cal_a16",), mech=("s2_cal_mech",)), tmp_path, boot=20)
    assert pilot["h1_holm"]["bool"]["family"] == "observed" and pilot["h1_holm"]["bool"]["m"] == len(pilot["h1"])


def test_h10a_reports_holm_over_the_32_registered_cells():
    from reach.analysis.s2 import cgp_confirm as G
    from test_cgp_confirm import _rows
    rows = (_rows("F-EML", 3.0, -0.5, seed=1) + _rows("BOOL-NAND-n5", 3.0, -0.4, seed=2)
            + _rows("BOOL-NOR-n5", -6.0, 0.5, seed=3) + _rows("BOOL-AON-n5", 0, 0, saturated=True))
    res = G.h10a(rows, boot=200)
    h = res["holm"]
    assert h["real"]["m"] == 14 and h["bool"]["m"] == 18 and len(G.declared_h10a_cells()) == 32
    assert len(h["real"]["missing"]) == 12 and len(h["bool"]["missing"]) == 12
    cells = {(c["set"], c["budget"]): c for c in res["cells"]}
    assert cells[("BOOL-AON-n5", 1)]["holm_status"] == "saturated (untestable)"
    assert not cells[("BOOL-NOR-n5", 1)]["holm_supported"]
    for c in res["cells"]:
        assert not c["holm_supported"] or c["supports"]
    assert res["summary"]["bool"] == dict(cells=6, supported=2, untestable=2)            # existing output kept
    obs = G.h10a(rows, boot=50, declared=None)["holm"]
    assert obs["real"]["m"] == 2 and obs["bool"]["m"] == 6


def test_h1_and_h10a_share_one_holm_function(tmp_path, monkeypatch):
    from reach.analysis.s2 import cgp_confirm as G
    from reach.analysis.s2 import confirm as C
    from test_cgp_confirm import _rows
    calls, adj = [], []
    real, real_holm = L.holm_cells, contrasts.holm
    monkeypatch.setattr(L, "holm_cells", lambda *a, **k: calls.append(k.get("alpha", a[2:3])) or real(*a, **k))
    monkeypatch.setattr(contrasts, "holm", lambda *a, **k: adj.append(1) or real_holm(*a, **k))
    _calibration(tmp_path)
    C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=20, allow_incomplete=True)
    G.h10a(_rows("F-EML", 3.0, -0.5, levels=(5, 9, 13), n_targets=6, seeds=2), boot=20)
    assert calls == [(0.05,)] * 3 and len(adj) >= 3        # H1 and H9a (confirm; D078), H10a; Holm per domain

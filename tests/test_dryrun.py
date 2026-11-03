"""DRYRUN driver (REQ-ANA-09) on simulated pilot phases with a known law: every section runs and recovers it."""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import dryrun as D

SETS = ("BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5")


def _rows(g, algo, c, seed, kappa=0.6, alpha=0.5):
    rng = np.random.default_rng(seed)
    out = []
    for K in range(6, 28, 2):
        for t in range(8):
            for s in range(3):
                T = (-math.log(rng.random()) / math.exp(c - alpha * K * math.log(2))) ** (1 / kappa)
                ge = math.ceil(T) - 1
                ok = ge < 800
                out.append(dict(key=f"{g}|{algo}|{K}|{t}|{s}", set=g, target_id=f"{g}_K{K}_{t}", seed=s,
                                K_construction=K, K_proven=K, algorithm=algo, config="base", pop_size=100,
                                units=dict(nodes=float(K), bits_uniform=2.0 * K, bits_syn=1.5 * K, bits_sem=float(K)),
                                generations=800 if algo == "A" else 200, gen_exact=ge if ok else None,
                                gens_reached=ge if ok else 800, terminal_state="converged" if ok else "budget_exhausted",
                                cpu_s=1.0))
    return [r for r in out if algo == "A" or r["gen_exact"] is None or r["gen_exact"] < 200]


def _write(root, phase, rows):
    d = root / "data/runs" / phase
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(n_tasks=len(rows), tasks=[])))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_dryrun_runs_every_section_and_recovers_the_law(tmp_path):
    _write(tmp_path, "s2_pilot_reach_a", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)])
    _write(tmp_path, "s2_pilot_mech", _rows("BOOL-NAND-n5", "B", 0.5, 9))
    res = D.build(tmp_path, delta=0.01, boot=40)
    assert all(c["hi"] < 0 for c in res["h1"])                                   # H1 holds in every cell
    per = res["h4"]["bool"]["per_set"]["BOOL-NAND-n5"]["nodes"]
    assert per["rate"] == pytest.approx(0.6 / 0.5, abs=0.4)                      # kappa / alpha nodes per doubling
    assert res["h4"]["bool"]["law_nodes"]["beta"] > 0
    assert res["h5"]["BOOL-NAND-n5"]["A-B"]["diff"] > 0                          # A beats the weaker regime
    assert set(res["h2"]["bool"]["units"]) == set(D.UNITS)
    text = D.render(res)
    assert text.startswith("# DRYRUN") and "never quote" in text


def test_calibration_uses_the_registered_constants_and_phases(tmp_path):
    """The confirmatory path reads the s2_cal_* reach phases and the Δ filed in Stage 1, never pilot-derived values."""
    from reach.analysis.s2 import confirm as C
    assert C.REGISTERED["delta"] == {"real": 0.0098, "bool": 0.0059} and C.REGISTERED["boot"] == 2000
    assert C.PHASE_MAPS["calibration"]["reach"] == ("s2_cal_a0", "s2_cal_a16")
    _write(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)])
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=30, allow_incomplete=True)
    assert res["h2"]["bool"]["delta"] == 0.0059 and res["label"].startswith("CALIBRATION")
    h2 = res["h2"]["bool"]                      # the preferred unit follows the registered rule (D079 a; row 97)
    from reach.analysis.s2.invariance import preferred_unit
    assert h2["preferred"] == preferred_unit(h2["units"], C.REGISTERED["alpha"])["preferred"]
    assert h2["report_unit"] in C.UNITS and "deviating" in h2


def test_killed_runs_count_as_failures_and_h7_sees_the_budget(tmp_path):
    """Review findings 2 and 4: a run killed without a result has no units in its row; it must still enter as a failure
    (primary rule), with its units from the target file. H7 is selected on records at several budgets."""
    from reach.analysis.s2 import confirm as C
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    for r in rows:
        r["generations"] = 3200
    _write(tmp_path, "s2_cal_a16", rows + [dict(key="killed", set="BOOL-NAND-n5", target_id="BOOL-NAND-n5_K6_0", seed=99,
                                                K=6, K_proven=6, algorithm="A", config="base", pop_size=100,
                                                generations=3200, terminal_state="timeout")])
    for g in SETS:
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        (d / "calibration.json").write_text(json.dumps([dict(target_id=r["target_id"], units=r["units"]) for r in rows
                                                        if r["set"] == g]))
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=20, allow_incomplete=True)
    cell = next(c for c in res["h1"] if c["set"] == "BOOL-NAND-n5" and c["budget"] == 1)
    n_runs = sum(1 for r in rows if r["set"] == "BOOL-NAND-n5")
    assert cell["n"] == n_runs + 1                                         # the killed run is in, as a failure
    assert res["h7"]["bool"]["budget_term"] in (True, False) and res["h7"]["bool"]["budgets"] == list(C.H4_BUDGETS)


def test_confirmatory_analysis_survives_degenerate_cells(tmp_path):
    """Guardrail row 63: a set where regime B never succeeds, plus runs killed without a result, must give
    not-estimable flags, never a crash (Stage 1: saturated cells are reported, not failures)."""
    from reach.analysis.s2 import confirm as C
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    for r in rows:
        r["generations"] = 3200
    _write(tmp_path, "s2_cal_a16", rows)
    b = [dict(r, algorithm="B", generations=200, gen_exact=None, gens_reached=200, terminal_state="budget_exhausted",
              key=r["key"] + "|B") for r in rows if r["set"] == "BOOL-NAND-n5"]
    b += [dict(key="kill|B", set="BOOL-NAND-n5", target_id=b[0]["target_id"], seed=0, K=6, algorithm="B",
               config="base", pop_size=100, generations=200, terminal_state="timeout")]
    _write(tmp_path, "s2_cal_mech", b)
    for g in SETS:
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        (d / "calibration.json").write_text(json.dumps([dict(target_id=r["target_id"], units=r["units"]) for r in rows
                                                        if r["set"] == g]))
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=20, allow_incomplete=True)
    ab = res["h5"]["BOOL-NAND-n5"]["A-B"]
    assert not ab["estimable"] and math.isnan(ab["lo"])
    assert any(c["algorithm"] == "B" for c in res["h1"])                 # the saturated B cell is reported


def test_lower_bound_sizes_move_only_constructed_targets_and_respect_parity():
    """Sensitivity (exploratory, 2026-10-25): a constructed target's true minimum size lies in (ceiling, K]; the lower
    bound is the smallest size above the ceiling the set can build (odd only for binary-only sets)."""
    from reach.analysis.s2 import confirm as C
    recs = [dict(set="S", target_id="a", K=9, units=dict(nodes=9.0, bits_sem=3.0)),
            dict(set="S", target_id="b", K=41, units=dict(nodes=41.0, bits_sem=30.0))]
    units = lambda K: dict(nodes=float(K), bits_uniform=2.0 * K, bits_syn=1.5 * K, bits_sem=float(K))   # noqa: E731
    out = C.lower_bound_sizes(recs, constructed={("S", "b")}, ceilings={"S": 31}, parities={"S": {1}},
                              units_fn={"S": units})
    assert out[0]["K"] == 9 and out[0]["units"]["bits_sem"] == 3.0                 # proven: unchanged
    assert out[1]["K"] == 33 and out[1]["units"]["nodes"] == 33.0 and out[1]["units"]["bits_sem"] == 33.0
    assert recs[1]["K"] == 41                                                      # the input is not modified

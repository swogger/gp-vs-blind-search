"""Descriptive analyses for the paper's Figure 2 and Section 4.5 (D111; REQ-ANA-97, REQ-ANA-98): one-run reach per
budget on the held-out real sets, and the exact null's reach read with GP's own estimator."""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import heldout_reach as hr
from reach.paths import ROOT


def _row(target, gen_exact, generations=3200, **kw):
    r = dict(set="R-X", target_id=target, seed=1, algorithm="A", config="base", K=5, K_proven=5,
             units={"bits_sem": 7.0}, pop_size=100, generations=generations, gens_reached=generations,
             terminal_state="converged" if gen_exact is not None else "budget_exhausted", gen_exact=gen_exact,
             gen_tol=gen_exact)
    r.update(kw)
    return r


def test_null_under_the_estimator_sits_between_the_last_full_level_and_the_next():
    # three levels with 2, 8 and 36 functions; E = 12 fills levels 1 and 3 and 2 of 36 at level 5
    U = {1: 2, 3: 8, 5: 36}
    bits = {k: math.log2(sum(U[j] for j in U if j <= k)) for k in U}
    targets = [dict(K=k, bits_sem=bits[k]) for k in U for _ in range(3)]
    k = hr.null_k50_under_estimator(targets, U, E=12)
    assert bits[3] < k <= bits[5]                         # found everything to level 3, almost nothing at level 5
    assert k > math.log2(12)                               # the estimator reads the null above its exact reach
    assert hr.null_k50_under_estimator(targets, U, E=46) > bits[5] - 1e-9 or np.isnan(
        hr.null_k50_under_estimator(targets, U, E=46))     # every level full: no 50 percent crossing


def test_one_run_curve_reads_prefixes_and_is_monotone():
    rows = []
    for i, g in enumerate((10, 150, 350, 700, 1500, 3100, None, None)):
        rows.append(_row(f"t{i}", g, units={"bits_sem": 4.0 + i}))
    rows.append(_row("t9", 20, generations=200, units={"bits_sem": 3.0}))   # a 1 x E0 run: budget 1 only
    recs = hr.tidy_rows(rows, (1, 2, 4, 8, 16))
    assert {r["budget"] for r in recs if r["target_id"] == "t9"} == {1}
    curve = hr.k50_by_budget(recs)
    ks = [curve[b] for b in (1, 2, 4, 8, 16)]
    assert all(np.isfinite(ks)) and all(a <= b + 1e-9 for a, b in zip(ks, ks[1:]))


@pytest.mark.study_machine
def test_recorded_one_run_reach_matches_the_remedy_report_at_its_budgets():
    out = hr.build(ROOT)
    rem = json.load(open(ROOT / "data/analysis/remedy.json"))
    for s in hr.SETS:
        for b, key in ((1, "tree GP, 1 x E0"), (16, "tree GP, 16 x E0")):
            assert abs(out["one_run"][s][str(b)]["k50"] - rem["systems"][key][s]["k50"]) < 1e-6, (s, b)
    for s in hr.SETS:
        v = out["null_estimator"][s]
        assert np.isfinite(v["k50_enum"]) and v["offset"] > 0, s
        assert abs(v["log2E"] - math.log2(320000)) < 1e-12


@pytest.mark.study_machine
def test_calibration_offsets_keep_every_sign_and_match_the_registered_gaps():
    out = hr.build(ROOT)
    cal = out["calibration_offsets"]
    rem = json.load(open(ROOT / "data/analysis/h6d.json"))["domain"]
    assert abs(cal["real_mean"]["gap_exact"] - rem["real"]["gap"]) < 1e-6
    assert abs(cal["bool_mean"]["gap_exact"] - rem["bool"]["gap"]) < 1e-6
    assert cal["real_mean"]["signs_agree"] and cal["bool_mean"]["signs_agree"]
    for s in hr.CAL_REAL + hr.CAL_BOOL:
        assert np.isfinite(cal[s]["k50_enum"]), s


@pytest.mark.study_machine
def test_null_by_budget_is_finite_and_matches_the_single_budget_reading_at_16():
    out = hr.build(ROOT)
    nb = out["null_estimator_by_budget"]
    assert [int(b) for b in nb] == list(hr.BUDGETS)
    assert all(np.isfinite(v["mean"]) for v in nb.values())
    for s in hr.SETS:
        assert abs(nb["16"]["per_set"][s] - out["null_estimator"][s]["k50_enum"]) < 1e-9, s

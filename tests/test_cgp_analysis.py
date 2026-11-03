"""CGP validation and pilot analyses (D072 step 3; docs/design/cgp.md), written before the code and before any
validation or pilot data exist: REQ-ANA-23, REQ-ANA-24. Synthetic records only."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from reach.analysis.s2 import cgp_pilot as CP
from reach.analysis.s2 import cgp_validate as CV

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ validation (REQ-ANA-23)
def test_reference_values_are_the_published_ones(tmp_path):
    ref = CV.reference_values(ROOT)
    assert len(ref) == 50 and np.median(ref) == 1214 and min(ref) == 214 and max(ref) == 3921
    src = json.loads((ROOT / CV.REF_DIR / "source.json").read_text())
    assert "brianwgoldman/Analysis-of-CGPs-Mechanisms" in src["url"] and len(src["files"]["rdata.csv"]) == 64
    # planted violation: a changed reference file is refused
    d = tmp_path / CV.REF_DIR
    shutil.copytree(ROOT / CV.REF_DIR, d)
    p = d / "rdata.csv"
    p.write_text(p.read_text().replace("parity,single,normal,500,1,901,", "parity,single,normal,500,1,901,1"))
    with pytest.raises(ValueError, match="SHA-256"):
        CV.reference_values(tmp_path)


def _vrows(counts, n_nodes, start_seed=0, **over):
    """Synthetic validation records: offspring-level count c is recorded as eval_index_exact = c + 1."""
    rows = []
    for i, c in enumerate(counts):
        r = dict(algorithm="CGP", set="BOOL-AONN-n3", n_nodes=n_nodes, seed=start_seed + i, exact=True,
                 eval_index_exact=int(c) + 1, evals_exact=int(c) + 4, terminal_state="converged", active_nodes=39,
                 cpu_s=0.3, claim_rejected=False, missed_exact=False)
        r.update(over)
        rows.append(r)
    return rows


def _like_reference(ref, n=200, scale=1.0, seed=1):
    rng = np.random.default_rng(seed)
    return np.round(rng.choice(np.asarray(ref, float), size=n, replace=True) * scale)


def _passing(ref):
    return _vrows(_like_reference(ref), 500) + _vrows(_like_reference(ref, scale=1358 / 1214, seed=2), 3000)


def test_validation_passes_on_data_like_the_reference():
    ref = CV.reference_values(ROOT)
    res = CV.assess(_passing(ref), ref)
    assert res["verdict"] == "PASS", res
    s = res["settings"]["500"]
    assert s["all_succeed"] and s["median_ok"] and s["ks_p"] >= 0.01 and 885 <= s["median"] <= 1487
    # offspring-level: eval_index_exact - 1 (Goldman and Punch count the initial parent as 0)
    one = CV.assess(_vrows(ref * 4, 500) + _vrows(_like_reference(ref, scale=1358 / 1214), 3000), ref)
    assert one["settings"]["500"]["median"] == 1214 and one["settings"]["500"]["ks_p"] > 0.99


def test_validation_fails_on_planted_failures():
    ref = CV.reference_values(ROOT)
    good = _passing(ref)
    # one run failed: "every run succeeds" is violated
    bad = [dict(r) for r in good]
    bad[3].update(exact=False, eval_index_exact=None, evals_exact=None, terminal_state="budget_exhausted")
    res = CV.assess(bad, ref)
    assert res["verdict"] == "FAIL" and not res["settings"]["500"]["all_succeed"]
    # a CPU-killed row (no result) is not a success either
    bad = [dict(r) for r in good]
    bad[0] = dict(algorithm="CGP", n_nodes=500, seed=0, terminal_state="timeout")
    assert CV.assess(bad, ref)["verdict"] == "FAIL"
    # twice as many evaluations: median outside the CI and KS rejects
    res = CV.assess(_vrows(_like_reference(ref, scale=2.0), 500) + good[200:], ref)
    assert res["verdict"] == "FAIL" and not res["settings"]["500"]["median_ok"] and res["settings"]["500"]["ks_p"] < 0.01
    # same median, wrong shape: the median passes, KS rejects
    narrow = np.round(np.linspace(1150, 1280, 200))
    res = CV.assess(_vrows(narrow, 500) + good[200:], ref)
    assert res["settings"]["500"]["median_ok"] and res["settings"]["500"]["ks_p"] < 0.01 and res["verdict"] == "FAIL"
    # 3,000 nodes: median beyond +25% of 1,358
    res = CV.assess(good[:200] + _vrows(_like_reference(ref, scale=1.5 * 1358 / 1214), 3000), ref)
    assert res["verdict"] == "FAIL" and not res["settings"]["3000"]["median_ok"] and res["settings"]["500"]["passed"]
    # an incomplete phase never passes
    res = CV.assess(good[:150] + good[200:], ref)
    assert res["verdict"] == "INCOMPLETE" and not res["settings"]["500"]["complete"]
    assert CV.assess([], ref)["verdict"] == "INCOMPLETE"
    # ... but a failed run fails it at once, finished or not
    bad = [dict(r) for r in good[:150] + good[200:]]
    bad[0].update(exact=False, eval_index_exact=None, terminal_state="budget_exhausted")
    assert CV.assess(bad, ref)["verdict"] == "FAIL"


def test_validation_report_is_written(tmp_path):
    ref_dir = tmp_path / CV.REF_DIR
    shutil.copytree(ROOT / CV.REF_DIR, ref_dir)
    ph = tmp_path / "data/runs" / CV.PHASE
    ph.mkdir(parents=True)
    rows = _passing(CV.reference_values(ROOT))
    for i, r in enumerate(rows):
        r["key"] = f"k{i}"
    (ph / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = CV.main(tmp_path)
    text = (tmp_path / CV.REPORT).read_text()
    assert res["verdict"] == "PASS" and "PASS" in text and "1,214" in text and len(text.splitlines()) <= 150
    assert json.loads((tmp_path / CV.JSON_OUT).read_text())["verdict"] == "PASS"


# ------------------------------------------------------------------ pilot (REQ-ANA-24)
def _prow(g, n, tid, seed, evals_exact=None, cpu=1.0, state=None, reached=320_000, **over):
    r = dict(algorithm="CGP", set=g, n_nodes=n, config=f"n{n}", target_id=tid, seed=seed, budget=16, budget_E=320_000,
             exact=evals_exact is not None, evals_exact=evals_exact, evals_reached=reached, cpu_s=cpu,
             terminal_state=state or ("converged" if evals_exact is not None else "budget_exhausted"),
             claim_rejected=False, missed_exact=False, K_construction=5, K_proven=5)
    r.update(over)
    return r


E0 = 20_000


def test_pilot_rule_averages_budgets_and_weighs_sets_equally():
    rows = []
    # real: F-EML has 1 target, R-ELEM4 has 9 (sets weigh equally, not runs)
    for n, (fe, re) in {100: (1 * E0, None), 500: (None, 1 * E0), 2000: (16 * E0, 16 * E0)}.items():
        rows.append(_prow("F-EML", n, "fe0", 1, fe))
        rows += [_prow("R-ELEM4", n, f"re{i}", 1, re) for i in range(9)]
    # bool: every length solves everything at 1/4 x E0
    rows += [_prow(g, n, "b0", 1, E0 // 4) for g in ("BOOL-NAND-n5", "BOOL-AONX-n5") for n in (100, 500, 2000)]
    res = CP.assess(rows)
    sc = res["scores"]["real"]
    # success at budgets (1/4, 1, 4, 16): solved at 1xE0 -> (0,1,1,1) = 0.75; at 16xE0 -> (0,0,0,1) = 0.25
    assert sc[100] == pytest.approx((0.75 + 0) / 2) and sc[500] == pytest.approx((0 + 0.75) / 2)
    assert sc[2000] == pytest.approx(0.25)
    assert res["choice"]["real"] == 100                       # 100 and 500 tie (0.375): the shorter genome
    assert res["choice"]["bool"] == 100 and res["scores"]["bool"][2000] == 1.0
    # a CPU stop before a budget counts as a failure there (load.tidy, censored as failure)
    rows2 = [r for r in rows if r["set"] != "F-EML"] + [
        _prow("F-EML", 100, "fe0", 1, None, state="timeout", reached=2 * E0),
        _prow("F-EML", 500, "fe0", 1, None), _prow("F-EML", 2000, "fe0", 1, 16 * E0)]
    assert CP.assess(rows2)["scores"]["real"][100] == pytest.approx(0)


def test_pilot_rule_gives_near_ties_to_the_shorter_genome():
    assert CP.choose({100: 0.50, 500: 0.515, 2000: 0.53}) == 500      # 0.015 below the best: shorter wins
    assert CP.choose({100: 0.50, 500: 0.51, 2000: 0.52}) == 500         # 100 is 0.02 below: not a near tie; 500 is
    assert CP.choose({100: 0.40, 500: 0.45, 2000: 0.60}) == 2000
    assert CP.choose({100: 0.60, 500: 0.70, 2000: 0.60}) == 500
    assert CP.choose({100: 0.70, 500: 0.70, 2000: 0.70}) == 100
    assert CP.choose({100: 0.48, 500: 0.50, 2000: 0.4999}) == 500       # exactly 0.02 below: not a near tie
    assert CP.choose({100: float("nan"), 500: 0.5, 2000: 0.4}) is None  # a missing cell: no choice


def test_pilot_costs_and_agreement_per_setting():
    rows = [_prow("F-EML", 100, f"t{i}", 1, None, cpu=float(i + 1)) for i in range(100)]
    rows[5].update(terminal_state="timeout", evals_reached=E0)
    rows[6] = dict(algorithm="CGP", set="F-EML", n_nodes=100, target_id="t6", seed=1, terminal_state="timeout",
                   cpu_s=1900.0)                                         # killed at the hard limit: no result
    rows[7].update(claim_rejected=True, terminal_state="early_stopped", evals_reached=E0)
    rows[8].update(missed_exact=True)
    rows.append(_prow("F-EML", 100, "e", 1, None, state="error"))
    c = CP.costs(rows)[("F-EML", 100)]
    assert c["runs"] == 101 and c["errors"] == 1 and c["cpu_limit"] == 2 and c["claim_rejected"] == 1
    assert c["missed_exact"] == 1
    cpu = [float(i + 1) for i in range(100)]
    cpu[6] = 1900.0
    assert c["cpu_mean"] == pytest.approx(np.mean(cpu)) and c["cpu_p99"] == pytest.approx(np.percentile(cpu, 99))
    assert c["cpu_max"] == 1900.0


def test_pilot_report_is_written(tmp_path):
    ph = tmp_path / "data/runs" / CP.PHASE
    ph.mkdir(parents=True)
    rows = [_prow(g, n, "t", 1, E0) for g in CP.SETS["real"] + CP.SETS["bool"] for n in CP.LENGTHS]
    for i, r in enumerate(rows):
        r["key"] = f"k{i}"
    (ph / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ph / "manifest.json").write_text(json.dumps(dict(n_tasks=len(rows) + 5, tasks=[])))
    res = CP.main(tmp_path)
    text = (tmp_path / CP.REPORT).read_text()
    assert res["complete"] is False and "provisional" in text.lower() and len(text.splitlines()) <= 150
    assert res["choice"] == {"real": 100, "bool": 100}
    assert json.loads((tmp_path / CP.JSON_OUT).read_text())["choice"] == {"real": 100, "bool": 100}


def test_rejected_claims_are_failures_and_reported(tmp_path):
    """A claim Python rejects ends the run as early_stopped (cgp.outcome): it is a failure at every budget, never
    dropped, and the number of rejections is reported per setting (docs/design/cgp.md, "Rejected claims")."""
    ref = CV.reference_values(ROOT)
    rows = _passing(ref)
    rows[10].update(exact=False, claim_rejected=True, terminal_state="early_stopped", evals_exact=None,
                    eval_index_exact=None, evals_reached=900)
    res = CV.assess(rows, ref)
    s = res["settings"]["500"]
    assert res["verdict"] == "FAIL" and s["claim_rejected"] == 1 and s["successes"] == 199 and s["rows"] == 200
    assert "| claims rejected by Python | 1 | 0 |" in CV.report(res)
    # pilot: a rejected claim at E0 is a failure at 1/4 x E0 too (it never claimed there) and at every later budget
    prow = [_prow(g, n, "t", 1, E0 // 4) for g in CP.SETS["real"] + CP.SETS["bool"] for n in CP.LENGTHS]
    prow[0].update(exact=False, evals_exact=None, claim_rejected=True, terminal_state="early_stopped",
                   evals_reached=E0)
    res = CP.assess(prow)
    assert res["recovery"][f"{prow[0]['set']}|{prow[0]['n_nodes']}"] == {b: 0.0 for b in CP.BUDGETS}
    cost = next(c for c in res["costs"] if (c["set"], c["length"]) == (prow[0]["set"], prow[0]["n_nodes"]))
    assert cost["claim_rejected"] == 1 and "| 1 | 0 |" in CP.report(res)

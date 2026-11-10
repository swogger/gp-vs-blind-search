"""Exploratory revision analyses (D125; REQ-ANA-112, REQ-ANA-113): the estimators on synthetic data, the gzip reader,
and the recorded build, which reproduces the deposited registered values and the audit's figures."""
import gzip
import json
import math

import pytest

from reach.analysis.s2 import ert, revision_report as R
from reach.paths import ROOT


def _tree(gen_exact, generations=200, gens_reached=None, state=None, **kw):
    r = dict(pop_size=100, generations=generations, gens_reached=generations if gens_reached is None else gens_reached,
             gen_exact=gen_exact, terminal_state=state or ("converged" if gen_exact is not None else "budget_exhausted"))
    r.update(kw)
    return r


def test_ert_counts_failures_and_censored_runs_at_the_budget():
    rows = [_tree(4),                                       # success: (4 + 1) x 100 = 500 evaluations
            _tree(None),                                    # failure: 20,000
            _tree(None, gens_reached=50, state="timeout"),  # stopped on its CPU limit: a failure charged E
            _tree(250, generations=3200),                   # a success after E0: a failure at E0, 20,000
            _tree(None, state="error"),                     # error: left out
            {k: v for k, v in _tree(None).items() if k != "gens_reached"}]   # killed without a result: failure
    e = ert.ert(rows, 20_000)
    assert (e["successes"], e["runs"], e["evaluations"]) == (1, 5, 500 + 4 * 20_000)
    assert e["ert"] == 80_500
    assert e["ert"] != 500                                  # a plant counting the successes only would give 500
    # at 16 x E0 the 200-generation rows are 'beyond' and left out; the long run succeeds at 25,100; the row killed
    # without a result is censored at every budget (`load.outcome`), so a failure charged 320,000
    assert ert.ert(rows, 320_000) == dict(ert=345_100, successes=1, runs=2, evaluations=345_100, lower_bound=None)
    none = ert.ert([_tree(None)] * 3, 20_000)
    assert math.isinf(none["ert"]) and none["lower_bound"] == 60_000
    cgp = [dict(budget_E=320_000, evals_reached=1041, evals_exact=1041, exact=True, terminal_state="converged"),
           dict(budget_E=320_000, evals_reached=320_000, evals_exact=None, exact=False, terminal_state="budget_exhausted")]
    assert ert.ert(cgp, 20_000, "cgp")["ert"] == 1041 + 20_000
    assert ert.gap_bits(1024, 256) == 2.0 and ert.gap_bits(10, float("inf")) == float("-inf")


def test_null_evaluations_by_level():
    U = {1: 2, 3: 8, 5: 36}
    n = ert.null_evals(U, 5)
    assert (n["cover"], n["expected"], n["below"]) == (46, 10 + 37 / 2, 10)
    assert ert.null_success(U, 5, 12) == 2 / 36 and ert.null_success(U, 3, 12) == 1.0
    assert ert.null_success(U, 5, 5) == 0.0
    # the expected rank is the mean position of a uniformly drawn target of the level
    assert n["expected"] == sum(10 + i for i in range(1, 37)) / 36


def test_read_phase_reads_the_committed_gzip_when_the_raw_file_is_absent(tmp_path):
    """Guardrail row 179: s4_restart, s2_cal_cgp and s2_ho_cgp are committed as results.jsonl.gz only (D097), and
    `load.read_phase` returned no row for them in any clean checkout (remedy.build's CGP systems came out empty)."""
    from reach.analysis.s2.load import read_phase
    d = tmp_path / "data/runs/ph"
    d.mkdir(parents=True)
    lines = [json.dumps(dict(key="a", v=1)), json.dumps(dict(key="b", v=1)), json.dumps(dict(key="a", v=2)), '{"key": "c"']
    with gzip.open(d / "results.jsonl.gz", "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    assert {r["key"]: r["v"] for r in read_phase("ph", tmp_path)} == {"a": 2, "b": 1}
    assert read_phase("absent", tmp_path) == []
    (d / "results.jsonl").write_text(json.dumps(dict(key="z", v=9)) + "\n")
    assert [r["key"] for r in read_phase("ph", tmp_path)] == ["z"]          # the raw file wins when present


def test_rows_read_the_committed_gzip_when_the_raw_file_is_absent(tmp_path):
    d = tmp_path / "data/runs/ph"
    d.mkdir(parents=True)
    lines = [json.dumps(dict(key="a", v=1)), json.dumps(dict(key="b", v=1)), json.dumps(dict(key="a", v=2)), '{"key": "c"']
    with gzip.open(d / "results.jsonl.gz", "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    got = {r["key"]: r["v"] for r in R.rows(tmp_path, "ph")}
    assert got == {"a": 2, "b": 1}                           # last row per key; the truncated line ignored
    (d / "results.jsonl").write_text(json.dumps(dict(key="z", v=9)) + "\n")
    assert [r["key"] for r in R.rows(tmp_path, "ph")] == ["z"]   # the raw file wins when present


def test_coverage_within_uses_the_same_runs_at_both_prefixes():
    K = [k for k, u in R.level_counts("R-ARITH4").items() if u > 0]
    n = max(K) + 1

    def rv(**pref):
        return dict(prefixes={P: dict(lv=[v] * n) for P, v in pref.items()})
    runs = [dict(set="R-ARITH4", rv=rv(**{"10000": 1, "320000": 3})), dict(set="R-ARITH4", rv=rv(**{"10000": 2})),
            dict(set="R-ARITH4", rv=rv(**{"10000": 1, "320000": 5})), dict(set="R-X", rv=rv(**{"10000": 9}))]
    c = R.coverage_within(ROOT, runs)
    assert c["n_runs"] == 2 and c["early"] == len(K) and c["late"] == 4 * len(K) and c["ratio"] == 4
    assert c["all_runs_early"]["n"] == 3 and c["stopped_between"] == dict(n=1, mean_early=2 * len(K))


@pytest.fixture(scope="module")
def built():
    return R.build(ROOT)


@pytest.mark.study_machine
def test_recorded_values_reproduce_the_deposited_and_audit_values(built):
    """build() raises unless every instrument check reproduces a deposited registered value (preempt.json, h6d.json,
    h11.json, calibration.json's H6c and H8b, remedy.json's CGP and Operon gaps, stage2.json's benchmark rows); the
    values below are the private audit scripts' and the paper's Figure 4 values (paper/values.lock.json)."""
    r = built
    assert r["label"].startswith("EXPLORATORY")
    b = r["bool_capped"]
    assert round(b["mean_gap"], 2) == 7.52 and b["sets_above_top"] == 2
    assert {s for s, v in b["per_set"].items() if v["capped"]} == {"BOOL-NAND-n5", "BOOL-NOR-n5"}
    assert b["n_proven_levels_by_budget"]["BOOL-NOR-n5"]["16"] == 2
    g = r["g_proven_reestimated"]
    assert round(g["mean_g"], 2) == 15.52 and round(g["mean_g_registered_twin"], 2) == 15.61
    c = r["coverage_within"]
    assert (c["n_runs"], round(c["early"], 1), round(c["late"], 1), round(c["ratio"], 2)) == (423, 517.5, 1120.4, 2.17)
    lock = json.loads((ROOT / "paper/values.lock.json").read_text())
    assert f"{c['all_runs_early']['mean']:,.0f}" == lock["{{disc.small.one_restart:,.0f}}"]
    assert f"{c['all_runs_late']['mean']:,.0f}" == lock["{{disc.small.one_run:,.0f}}"]
    assert str(c["all_runs_early"]["n"]) == lock["{{disc.runs.restarts}}"]
    h = r["h11_common"]
    assert h["bracketed"] == 2 and [round(h[k], 2) for k in ("bracketed_gain_lo", "bracketed_gain_hi", "bound_lo",
                                                              "bound_hi")] == [4.58, 4.73, 1.96, 2.42]
    assert round(r["s12_same_targets"]["base"], 2) == -4.35
    assert round(r["s12_same_targets"]["configs"]["R2"], 2) == -1.50
    t = r["tolerance_added"]
    assert (t["total"], t["lower_five"], t["upper_five"], t["R-ARITH4"]["total"]) == (46, 27, 19, 14)
    w = r["twin"]
    assert [round(w[k], 3) for k in ("interp_real", "interp_bool", "interp_diff")] == [6.973, 6.577, -0.396]
    assert (w["extrapolated_count"], w["n_sets"]) == (11, 14)
    assert round(w["by_budget"]["320000"]["bool"], 2) == 9.05
    v = r["realisable_vs_restart"]
    assert (round(v["mean"], 2), v["sets_above"], v["n_sets"]) == (0.89, 4, 6)
    i = r["iso"]["e0"]
    assert round(i["real"]["mean_gap"], 2) == -4.51 and round(i["bool"]["mean_gap"], 2) == 8.40   # the notes' values
    assert math.isclose(i["real"]["mean_gap"], i["real"]["mean_gap_per_run"], abs_tol=1e-9)
    bn = r["bench"]
    assert (bn["real"]["below"], bn["real"]["above"], bn["real"]["tie"]) == (16, 2, 12)
    assert (bn["bool"]["below"], bn["bool"]["above"], bn["bool"]["tie"]) == (8, 15, 2)
    assert round(bn["real"]["median_gap_cover"], 2) == -4.40 and round(bn["bool"]["median_gap_cover"], 2) == 2.65
    o = r["operon"]
    assert round(o["real_gap_16"], 2) == -7.12 and o["facts"]["draws_once"] and o["facts"]["one_fit"]
    assert round(o["budget_use"]["default"]["eval_share"], 2) == 0.08
    assert round(o["budget_use"]["symbolic"]["eval_share"], 2) == 1.00


@pytest.mark.study_machine
def test_instrument_checks_fire_on_a_changed_deposited_value(monkeypatch):
    real_json = R._json

    def planted(root, name):
        d = real_json(root, name)
        if name == "preempt":
            d["bool_bounds"]["readings"]["proven_only"]["per_set"]["BOOL-NAND-n5"]["k50_e0"] += 1e-6
        return d
    monkeypatch.setattr(R, "_json", planted)
    with pytest.raises(RuntimeError, match="instrument check"):
        R.bool_capped(ROOT, R.main_sets(ROOT)["bool"])


@pytest.mark.study_machine
def test_committed_json_and_report_match_a_fresh_build(built):
    def strip(d):
        return {k: v for k, v in d.items() if k not in ("built", "cpu_seconds")}
    fresh = json.dumps(strip(built), indent=1, default=float)
    have = json.loads((ROOT / "data/analysis/revision_exploratory.json").read_text())
    assert json.dumps(strip(have), indent=1, default=float) == fresh
    md = (ROOT / "docs/status/revision_exploratory.md").read_text()
    assert md.splitlines()[2:] == R.render(dict(built, built=have["built"])).splitlines()[2:]
    assert len(md.splitlines()) <= 150 and "EXPLORATORY" in md

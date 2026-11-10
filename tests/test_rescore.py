"""Independent re-scoring of the held-out and restart phases (QA; D068 check 2 extended, D125; REQ-RUN-50; guardrail
row 171). Written before the code; each check is shown to fire on a planted violation."""
import hashlib
import json
from pathlib import Path

import pytest

from reach.runner import phases as P

ROOT = Path(__file__).resolve().parents[1]


def test_rescore_phase_covers_every_set_of_the_five_phases():
    """Guardrail rows 171 and 177: every held-out and restart phase whose exact runs enter a test or an output the
    paper reports, s2_ho_bench included."""
    tasks = P.PHASES["s2_audit_labels_ho"]["build"]()
    phases = ("s2_ho_a0", "s2_ho_a16", "s2_ho_bench", "s2_ho_operon", "s4_restart")
    assert tuple(P.RESCORE_PHASES) == phases
    want = {(ph, t["meta"]["set"]) for ph in phases for t in P._manifest_tasks(ph)}
    n_bench = len({t["meta"]["set"] for t in P._manifest_tasks("s2_ho_bench")})
    assert {(t["meta"]["phase"], t["meta"]["set"]) for t in tasks} == want and len(tasks) == len(want) == 32 + n_bench
    for t in tasks:
        a = t["argv"]
        assert "reach.tasks.s2_rescore" in a and a[a.index("--phase") + 1] == t["meta"]["phase"]
        n_sets = sum(1 for x in tasks if x["meta"]["phase"] == t["meta"]["phase"])
        assert a[a.index("--n_fail") + 1] == str(300 // n_sets)
    from reach import s2_lock as L
    assert L.registered_tasks_file("s2_audit_labels_ho") is None             # QA: not a registered phase


def _targets(tmp_path, g, ts, name="heldout"):
    d = tmp_path / f"data/targets_s2/{g}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(ts))
    return hashlib.sha256((d / f"{name}.json").read_bytes()).hexdigest()


def test_rescore_finds_planted_errors(tmp_path):
    from reach.tasks.s2_rescore import rescore, targets_of
    t = dict(target_id="R-ARITH_K07_00", expression="*(x, +(x, 1))", target_scale=12.0, K_construction=7)
    sha = _targets(tmp_path, "R-ARITH", [t])
    _targets(tmp_path, "R-ARITH", [dict(t, target_id="bench_x", expression="*(x, x)", target_scale=9.0)], "bench")
    targets, shas = targets_of("R-ARITH", tmp_path)
    assert shas["R-ARITH_K07_00"] == sha and ("R-ARITH", "bench_x") in targets
    base = dict(set="R-ARITH", target_id=t["target_id"], terminal_state="converged", exact=True, algorithm="A",
                targets_sha256=sha, final_tree_size=5)
    rows = [dict(base, key="ok", best_expr="+(*(x, x), x)"),
            dict(base, key="big", best_expr="", best_expr_full="+(x, *(x, x))"),
            dict(base, key="wrong", best_expr="*(x, x)"),                                  # planted: not the target
            dict(base, key="size", best_expr="", best_expr_full="+(x, *(x, x))", final_tree_size=7),
            dict(base, key="file", best_expr="+(*(x, x), x)", targets_sha256="0" * 64),
            dict(base, key="fail", exact=False, terminal_state="budget_exhausted", best_expr="+(x, x)")]
    out = rescore(rows, targets, shas, n_fail=5)
    assert out["exact_checked"] == 5 and out["exact_disputed"] == ["wrong"] and out["failures_checked"] == 1
    assert out["full_size_mismatch"] == ["size"] and out["target_file_mismatch"] == ["file"]
    assert out["unchecked"] == {} and out["exact_rows"] == 5 and out["rows"] == 6
    lost = rescore([dict(base, key="none", best_expr="")], targets, shas, n_fail=5)    # planted: no expression
    assert lost["unchecked"] == {"none": "no expression recorded"}


def _labels_rows(phase, g, exact, checked, **kw):
    return dict(dict(key=f"labels|{phase}|{g}", phase=phase, set=g, terminal_state="completed", exact_rows=exact,
                     exact_checked=checked, exact_disputed=[], failures_disputed=[], unchecked={},
                     exact_near_miss=[], exact_ieee_only=[], full_size_mismatch=[], target_file_mismatch=[],
                     failures_checked=3, failures_near=[]), **kw)


def test_heldout_labels_report_needs_every_exact_run(tmp_path):
    from reach.analysis.s2.audit import heldout_labels_report
    d = tmp_path / "data/runs/s2_audit_labels_ho"
    d.mkdir(parents=True)
    tasks = [dict(key=f"labels|{ph}|G", meta=dict(phase=ph, set="G")) for ph in ("p1", "p2")]

    def write(rows):
        (d / "manifest.json").write_text(json.dumps(dict(n_tasks=2, tasks=tasks)))
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    good = [_labels_rows("p1", "G", 10, 10), _labels_rows("p2", "G", 4, 4)]
    write(good)
    rep = heldout_labels_report(tmp_path)
    assert rep["ok"] and rep["phases"]["p1"]["exact_checked"] == 10
    for bad in ([good[0]], [good[0], _labels_rows("p2", "G", 4, 3)],
                [good[0], _labels_rows("p2", "G", 4, 4, exact_disputed=["k"])],
                [good[0], _labels_rows("p2", "G", 4, 4, full_size_mismatch=["k"])],
                [good[0], _labels_rows("p2", "G", 4, 4, unchecked={"k": "no expression recorded"})]):
        write(bad)
        assert not heldout_labels_report(tmp_path)["ok"], bad


@pytest.mark.study_machine
def test_rescore_agrees_on_a_sample_of_heldout_rows():
    """The instrument on known-good data: 15 exact s2_ho_a0 runs of two real and one Boolean held-out set (first by
    key) and 5 failures are all agreed."""
    from reach.tasks.s2_rescore import rescore, targets_of
    rows = P.phase_rows("s2_ho_a0")
    for g in ("R-ARITH", "RS-add_sqrt", "BOOL-AN-n5"):
        mine = sorted((r for r in rows.values() if r["set"] == g), key=lambda r: r["key"])
        pick = [r for r in mine if r["exact"]][:5] + [r for r in mine if not r["exact"]][:5]
        targets, shas = targets_of(g)
        out = rescore(pick, targets, shas, n_fail=5)
        assert out["exact_checked"] == 5 and not out["exact_disputed"] and not out["failures_disputed"], g
        assert not out["target_file_mismatch"] and not out["full_size_mismatch"], g


def test_rescore_phase_covers_stage5s_long_runs():
    """Guardrail row 209: every exact run of s5_long_extra (which enters H11-2L) is re-scored before the analysis, by a
    QA phase that is neither registered nor locked, with the same task form as s2_audit_labels_ho."""
    tasks = P.PHASES["s2_audit_labels_s5"]["build"]()
    sets = sorted({t["meta"]["set"] for t in P._manifest_tasks("s5_long_extra")})
    assert len(sets) == 6
    assert sorted((t["meta"]["phase"], t["meta"]["set"]) for t in tasks) == [("s5_long_extra", g) for g in sets]
    for t in tasks:
        a = t["argv"]
        assert "reach.tasks.s2_rescore" in a and a[a.index("--phase") + 1] == "s5_long_extra"
        assert a[a.index("--set") + 1] == t["meta"]["set"] and a[a.index("--n_fail") + 1] == str(300 // len(sets))
    from reach import s2_lock as L
    assert L.registered_tasks_file("s2_audit_labels_s5") is None             # QA: not registered, not locked

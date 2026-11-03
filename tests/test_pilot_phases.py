"""Pilot phases (T14 part; docs/plan/phases/phase3_pilot.md). REQ-RUN-08, REQ-RUN-10."""
import hashlib
import json

import pytest

from reach.runner import phases as P


def test_pilot_batches_cover_the_calibration_sets_once():
    from reach import s2_sets as S
    from reach.s2_lock import held_out
    a, b = set(P.PILOT_SETS_A), set(P.PILOT_SETS_B)
    assert not a & b and a | b == set(S.SETS) - held_out() - set(P.DROPPED)


def test_run_task_is_explicit_and_checks_the_lock(tmp_path, monkeypatch):
    t = dict(target_id="X_K05_00", expression="nand(x1, x2)", K_construction=5, K_proven=5)
    d = tmp_path / "data/targets_s2/BOOL-NAND-n5"
    d.mkdir(parents=True)
    (d / "pilot.json").write_text(json.dumps([t]))
    (d / "pilot.sha256").write_text(hashlib.sha256((d / "pilot.json").read_bytes()).hexdigest() + "  pilot.json\n")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    task = P._s2_task("reach", "BOOL-NAND-n5", "pilot", t, "A", "base", 11, 4)
    argv = task["argv"]
    for flag in ("--algorithm", "--config", "--penalty", "--stop_on", "--seed_leaves", "--generations", "--pop_size",
                 "--timeout", "--targets_sha256"):
        assert flag in argv
    assert argv[argv.index("--penalty") + 1] == "scaled" and argv[argv.index("--stop_on") + 1] == "exact"
    assert argv[argv.index("--generations") + 1] == "800" and argv[argv.index("--pop_size") + 1] == "100"
    assert task["cpu_limit"] > float(argv[argv.index("--timeout") + 1])
    from reach import s2_lock
    monkeypatch.setattr(s2_lock, "STAGE2_FILE", tmp_path / "not_recorded.txt")  # before Stage 2, whatever the repo holds
    if "R-ARITH" in s2_lock.held_out():                                           # (guardrail row 143)
        with pytest.raises(PermissionError):
            P._s2_task("reach", "R-ARITH", "pilot", t, "A", "base", 11, 4)
    r2 = P._s2_task("robust", "BOOL-NAND-n5", "pilot", t, "A", "R2", 11, 4)["argv"]
    assert r2[r2.index("--pop_size") + 1] == "1000" and r2[r2.index("--generations") + 1] == "80"

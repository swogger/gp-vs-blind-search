"""The run queue: every terminal state recorded, CPU limit enforced, resumable, drainable, code pinned."""
import io
import json
import signal
import uuid

import pytest

from reach.runner.queue import Phase, Queue, read_results


TOKEN = uuid.uuid4().hex[:12]        # this session's tag: another suite's tasks never match this one's (row 145)


def task(key, mode="ok", cpu_limit=10, cpu=0.1):
    return dict(key=key, argv=["{python}", "-W", f"ignore:{TOKEN}", "-m", "reach.runner.selftest_task", "--mode", mode,
                               "--cpu", str(cpu)], cpu_limit=cpu_limit, meta=dict(mode=mode))


def run(ph, **kw):
    out = io.StringIO()
    q = Queue(ph, workers=kw.pop("workers", 4), install_signals=False, out=out, **kw)
    code = q.run()
    return q, code, out.getvalue()


@pytest.fixture
def phase(tmp_path):
    return Phase("t", runs_dir=tmp_path)


def test_every_terminal_state_is_recorded(phase):
    phase.write_manifest([task("ok"), task("tmo", "timeout", cpu_limit=1),
                          task("err", "error"), task("nores", "noresult")])
    _, code, out = run(phase)
    rows = phase.results()
    assert code == 0 and len(rows) == 4
    assert rows["ok"]["terminal_state"] == "converged" and rows["ok"]["queue_status"] == "ok"
    assert rows["tmo"]["terminal_state"] == "timeout" and rows["tmo"]["queue_status"] == "cpu_limit"
    assert rows["tmo"]["signal"] in (signal.SIGXCPU, signal.SIGKILL)
    assert rows["tmo"]["cpu_s"] >= 0.9                      # killed by CPU time, not wall time
    assert rows["err"]["terminal_state"] == "error" and rows["err"]["exit"] != 0
    assert "deliberate failure" in rows["err"]["error_tail"]
    assert rows["nores"]["terminal_state"] == "error" and rows["nores"]["exit"] == 0
    for r in rows.values():                                 # termination fields on every row
        assert {"exit", "signal", "cpu_s", "wall_s", "cpu_limit", "code_hash"} <= set(r)
    assert (phase.logs / "err.log").exists() and not (phase.logs / "ok.log").exists()
    assert "ETA" in out and "4/4" in out


def test_resume_skips_done_and_reruns_the_rest(phase):
    phase.write_manifest([task(f"k{i}") for i in range(6)])
    q1, _, _ = run(phase, limit=2)
    assert q1.session_done == 2 and len(phase.results()) == 2
    q2, _, out = run(phase)
    assert q2.n_done_before == 2 and q2.session_done == 4
    lines = phase.results_path.read_text().splitlines()
    assert len(lines) == 6 and len({json.loads(l)["key"] for l in lines}) == 6
    q3, _, out = run(phase)
    assert q3.session_done == 0 and "0 to run" in out


def test_stop_file_drains(phase):
    phase.write_manifest([task(f"k{i}") for i in range(3)])
    phase.stop_path.write_text("")
    q, _, out = run(phase)
    assert q.session_done == 0 and q.draining and "STOP file found" in out


def test_errors_rerun_only_on_request(phase):
    phase.write_manifest([task("err", "error"), task("ok")])
    run(phase)
    q, _, _ = run(phase)
    assert q.session_done == 0
    q, _, _ = run(phase, retry_errors=True)
    assert q.session_done == 1 and phase.results()["err"]["terminal_state"] == "error"


def test_code_change_is_refused(phase):
    phase.write_manifest([task("a"), task("b")])
    phase.dir.mkdir(parents=True, exist_ok=True)
    phase.results_path.write_text(json.dumps(dict(key="a", terminal_state="converged",
                                                  code_hash="000000000000")) + "\n")
    with pytest.raises(SystemExit, match="changed since earlier results"):
        run(phase)
    q, _, _ = run(phase, allow_code_change=True)
    assert q.session_done == 1


def test_tasks_import_the_code_snapshot(phase):
    phase.write_manifest([task("a")])
    q, _, _ = run(phase)
    module_file = phase.results()["a"]["module_file"]
    assert str(phase.dir / "code" / q.code_hash) in module_file
    from reach.paths import ROOT
    assert phase.results()["a"]["root"] == str(ROOT)          # not the snapshot's parent


def test_truncated_last_line_is_ignored(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(dict(key="a", terminal_state="converged")) + '\n{"key": "b", "termin')
    assert list(read_results(p)) == ["a"]


def test_manifest_rejects_duplicates_and_missing_cpu_limit(phase):
    with pytest.raises(ValueError, match="duplicate"):
        phase.write_manifest([task("a"), task("a")])
    with pytest.raises(ValueError, match="cpu_limit"):
        phase.write_manifest([dict(key="a", argv=["true"])])


def test_limit_is_cpu_time_not_wall_time(phase):
    """Rule (author, 2026-10-19): run limits are CPU time. A task that sleeps 3 wall seconds with a 1-CPU-second
    limit must finish normally; a task that burns CPU with the same limit must be killed (positive control)."""
    phase.write_manifest([dict(key="sleeper", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "sleep",
                                                    "--cpu", "0.05", "--sleep", "3"], cpu_limit=1),
                          task("burner", "timeout", cpu_limit=1)])
    run(phase)
    rows = phase.results()
    assert rows["sleeper"]["terminal_state"] == "converged" and rows["sleeper"]["wall_s"] >= 3
    assert rows["sleeper"]["cpu_s"] < 1
    assert rows["burner"]["terminal_state"] == "timeout"


def test_external_kill_is_not_reported_as_a_cpu_timeout(phase):
    """Lesson 2: a SIGKILL before the CPU limit (memory pressure, a user) is an error with its signal, never a censoring
    by the CPU limit (found 2026-10-20: BOOL-RICH-n5 killed at 5,193 of 21,600 CPU-s was reported as a timeout)."""
    phase.write_manifest([task("killed", "selfkill", cpu_limit=60), task("tmo", "timeout", cpu_limit=1)])
    run(phase)
    rows = phase.results()
    assert rows["killed"]["terminal_state"] == "error" and rows["killed"]["queue_status"] == "killed_signal_9"
    assert rows["tmo"]["terminal_state"] == "timeout" and rows["tmo"]["queue_status"] == "cpu_limit"


def test_task_env_is_passed_but_cannot_unpin_the_code(phase):
    """Operon runs need a library path; a task may add variables, never override PYTHONPATH or REACH_ROOT."""
    t = dict(key="env", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "env"], cpu_limit=10,
             env=dict(REACH_TEST_ENV="seen", DYLD_FALLBACK_LIBRARY_PATH="/tmp/libs"))
    phase.write_manifest([t])
    run(phase)
    row = phase.results()["env"]
    assert row["env_value"] == "seen"
    assert row["dyld_value"] == "/tmp/libs"            # guardrail row 52: macOS strips DYLD_* through /bin/sh
    for bad in ("PYTHONPATH", "REACH_ROOT"):
        with pytest.raises(ValueError):
            phase.write_manifest([dict(t, env={bad: "/elsewhere"})])


def test_resume_reuses_the_pinned_snapshot_after_a_source_change(phase):
    """Review finding 1 (guardrail row 54): after src/reach changes, a phase resumes on its own pinned snapshot, so
    results never mix code versions and a reboot cannot stall it. A missing snapshot is still refused (test above)."""
    import shutil
    from reach.runner.queue import SRC
    phase.write_manifest([task("a"), task("b")])
    pinned = phase.dir / "code" / "aaaaaaaaaaaa"
    shutil.copytree(SRC / "reach", pinned / "reach", ignore=shutil.ignore_patterns("__pycache__"))
    phase.results_path.write_text(json.dumps(dict(key="a", terminal_state="converged", code_hash="aaaaaaaaaaaa")) + "\n")
    q, _, _ = run(phase)
    row = phase.results()["b"]
    assert q.session_done == 1 and row["code_hash"] == "aaaaaaaaaaaa"
    assert str(pinned) in row["module_file"]                     # the task imported the pinned code


def test_pinned_resume_records_the_git_version_of_the_pinned_code(phase):
    """Review 2, finding 7: rows of a pinned resume carry the earlier rows' git version, not the current HEAD."""
    import shutil
    from reach.runner.queue import SRC
    phase.write_manifest([task("a"), task("b")])
    pinned = phase.dir / "code" / "bbbbbbbbbbbb"
    shutil.copytree(SRC / "reach", pinned / "reach", ignore=shutil.ignore_patterns("__pycache__"))
    phase.results_path.write_text(json.dumps(dict(key="a", terminal_state="converged", code_hash="bbbbbbbbbbbb",
                                                  git="cafe123 (launch)")) + "\n")
    run(phase)
    assert phase.results()["b"]["git"] == "cafe123 (launch)"


# ------------------------------------------- guardrail row 145: concurrent suites must not block each other's queues
def test_another_suites_selftest_tasks_do_not_match_this_ones(monkeypatch):
    """Two test suites running at once (the dev worktree's pre-commit and the main worktree's pre-push, 2026-10-29)
    ran selftest tasks with the same command line, so one suite's live task made the other's queue refuse to start
    (row 138's check) and a test failed. Each session's tasks now carry its own tag; its own task still matches."""
    import reach.runner.queue as Q
    mine = task("a")
    other = [a.replace(TOKEN, "0" * len(TOKEN)) for a in mine["argv"]]            # the same task, another session
    monkeypatch.setattr(Q, "_processes", lambda: [(99, "/usr/bin/python3 " + " ".join(other[1:]))])
    assert other != mine["argv"] and Q.live_task_processes([mine]) == []
    monkeypatch.setattr(Q, "_processes", lambda: [(99, "/usr/bin/python3 " + " ".join(mine["argv"][1:]))])
    assert Q.live_task_processes([mine]) == [(99, "a")]


# ------------------------------------------- guardrail row 138: orphans of a hard-killed queue block the relaunch
def test_live_task_processes_match_a_task_by_its_whole_command(monkeypatch):
    """A process runs a task when its command line ends with the task's argv after the interpreter (the shell wrapper
    and the interpreter path differ); another seed or target is not a match."""
    import reach.runner.queue as Q
    tasks = [dict(key="k1", argv=["{python}", "-m", "reach.tasks.s2_run", "--set", "A", "--seed", "1"]),
             dict(key="k2", argv=["/abs/python", "-m", "reach.tasks.s2_operon", "--seed", "2"])]
    monkeypatch.setattr(Q, "_processes", lambda: [
        (11, "/usr/bin/python3 -m reach.tasks.s2_run --set A --seed 1"),
        (12, "/bin/sh -c ulimit -t 1000; exec /x/python -m reach.tasks.s2_operon --seed 2"),
        (13, "/usr/bin/python3 -m reach.tasks.s2_run --set A --seed 11"),
        (14, "python -m reach.tasks.s2_run --set B --seed 1")])
    assert Q.live_task_processes(tasks) == [(11, "k1"), (12, "k2")]


def test_relaunch_refuses_while_a_task_of_the_phase_still_runs(phase):
    """Audit 2026-10-28, finding 6: tasks run in their own session and survive a SIGKILL of the queue; a relaunch reran
    the same key into the same log ("w") beside the orphan. The relaunch now refuses (exit 3, no task started) while a
    process still runs one of the phase's remaining tasks, and starts normally once it has ended."""
    import subprocess
    import sys
    import reach.runner.queue as Q
    t = task("orph", cpu=0.1)
    phase.write_manifest([t])
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"] + t["argv"][1:],
                              start_new_session=True)
    try:
        for _ in range(50):                              # wait until ps shows its full command line
            if Q.live_task_processes([t]):
                break
            import time
            time.sleep(0.1)
        q, code, out = run(phase)
        assert code == 3 and "still running" in out and not phase.results()
        assert str(orphan.pid) in out
    finally:
        orphan.kill()
        orphan.wait()
    q, code, out = run(phase)
    assert code == 0 and phase.results()["orph"]["terminal_state"] != "error"

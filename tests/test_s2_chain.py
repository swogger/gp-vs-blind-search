"""Stage 2 chain (REQ-RUN-37, D097): the registered order, completeness and error counts, one rerun of error rows."""
import importlib.util
import json
import re

import pytest

from reach.paths import ROOT

spec = importlib.util.spec_from_file_location("s2_chain", ROOT / "scripts/s2_chain.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def test_order_is_the_registered_order():
    from reach.s2_lock import STAGE2_PHASES
    form = (ROOT / "docs/registration/osf/stage2_form.md").read_text()
    planned = re.search(r"Planned order: ([^;]+);", form).group(1).split(", ")
    assert list(C.ORDER) == planned and "s2_ho_cgp" not in C.ORDER and set(C.ORDER) <= set(STAGE2_PHASES)


def _phase(runs, name, n, rows):
    d = runs / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(phase=name, n_tasks=n, tasks=[])))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_state_counts_rows_and_errors(tmp_path):
    assert C.state("s2_ho_a0", tmp_path) == ("unplanned", 0, 0)
    _phase(tmp_path, "a", 3, [dict(key="k1", terminal_state="converged"), dict(key="k2", terminal_state="error")])
    assert C.state("a", tmp_path) == ("incomplete", 2, 1)
    _phase(tmp_path, "b", 2, [dict(key="k1", terminal_state="error"), dict(key="k2", terminal_state="timeout"),
                               dict(key="k1", terminal_state="converged")])          # the last row of a key counts
    assert C.state("b", tmp_path) == ("complete", 2, 0)


def test_chain_skips_runs_retries_once_and_stops(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(C, "ORDER", ("p1", "p2", "p3", "p4"))
    monkeypatch.setattr(C, "running", lambda p, runs: False)
    monkeypatch.setattr(C, "current_code", lambda: C.REGISTERED_CODE)                  # src/reach as registered
    _phase(tmp_path, "p1", 1, [dict(key="k", terminal_state="converged")])            # done: skipped
    _phase(tmp_path, "p2", 2, [dict(key="k", terminal_state="error"), dict(key="j", terminal_state="converged")])

    def run(p, runs, retry=False):                                                     # p3 completes when run;
        calls.append((p, retry))                                                       # p4 is drained: incomplete
        if p == "p3":
            _phase(runs, "p3", 1, [dict(key="k", terminal_state="converged")])
    monkeypatch.setattr(C, "run", run)
    assert C.chain(tmp_path) == 1                                                      # stops at p4
    assert calls == [("p2", True), ("p3", False), ("p4", False)]
    calls.clear()
    assert C.chain(tmp_path) == 1 and calls == [("p4", False)]                         # p2's rerun was once only
    (tmp_path / "p4").mkdir()
    (tmp_path / "p4/STOP").write_text("")
    calls.clear()
    assert C.chain(tmp_path) == 1 and calls == []                                      # a STOP file: no start


def test_registered_code_is_the_registered_commits(tmp_path):
    """REQ-RUN-38: the guard's constant is the code hash of src/reach at the commit the Stage 2 registration names."""
    import io
    import subprocess
    import tarfile
    from reach.runner.queue import code_hash
    rec = (ROOT / "data/registration/stage2_registration.txt").read_text()
    commit = re.search(r"Registered commit ([0-9a-f]{40})", rec).group(1)
    tar = subprocess.run(["git", "archive", commit, "src/reach"], cwd=ROOT, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(tmp_path)
    src = tmp_path / "src/reach"
    assert code_hash(src) == C.REGISTERED_CODE
    f = sorted(src.rglob("*.py"))[0]                                   # planted: one changed file, another hash
    f.write_text(f.read_text() + "\n")
    assert code_hash(src) != C.REGISTERED_CODE


def test_chain_launches_only_on_the_registered_code(tmp_path, monkeypatch):
    """REQ-RUN-38 (Phase 7b step 1, D098): a phase without rows snapshots src/reach at launch, so it starts only when
    src/reach is the registered code; a phase with rows resumes on its own snapshot, so its rows' code hash decides."""
    calls = []
    monkeypatch.setattr(C, "ORDER", ("p1",))
    monkeypatch.setattr(C, "running", lambda p, runs: False)
    monkeypatch.setattr(C, "run", lambda p, runs, retry=False: calls.append((p, retry)))
    monkeypatch.setattr(C, "current_code", lambda: "0" * 12)               # planted: src/reach changed
    assert C.chain(tmp_path) == 1 and calls == []                           # refused: nothing launched
    monkeypatch.setattr(C, "current_code", lambda: C.REGISTERED_CODE)
    assert C.chain(tmp_path) == 1 and calls == [("p1", False)]              # launched (the stub leaves it unplanned)
    calls.clear()
    _phase(tmp_path, "p1", 2, [dict(key="k", terminal_state="error", code_hash="1" * 12)])   # rows on other code
    assert C.chain(tmp_path) == 1 and calls == []
    rows = [dict(key="k", terminal_state="error", code_hash=C.REGISTERED_CODE),
            dict(key="j", terminal_state="converged", code_hash=C.REGISTERED_CODE)]
    (tmp_path / "p1/results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(C, "current_code", lambda: "0" * 12)               # src/reach changed after the launch:
    assert C.chain(tmp_path) == 0 and calls == [("p1", True)]               # the error rerun resumes on the snapshot


def test_running_needs_the_phase_queue_not_just_a_live_pid(tmp_path):
    """Guardrail row 144: the pid file of a finished queue can name a pid the system has reused; a live pid alone made a
    restarted chain wait on a finished phase for as long as the unrelated process lived."""
    import os
    import subprocess
    import sys
    import time
    (tmp_path / "p1").mkdir()
    (tmp_path / "p").mkdir()
    (tmp_path / "p1/queue.pid").write_text(f"{os.getpid()}\n")             # planted: alive, but not a queue
    assert not C.running("p1", tmp_path)
    pad = "x" * 120                          # a command line far past 80 columns (procps truncates there)
    q = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", pad, "-m", "reach.runner", "run", "p1"])
    try:
        (tmp_path / "p1/queue.pid").write_text(f"{q.pid}\n")
        (tmp_path / "p/queue.pid").write_text(f"{q.pid}\n")
        deadline = time.time() + 10
        while not C.running("p1", tmp_path) and time.time() < deadline:   # until the child has exec'd
            time.sleep(0.05)
        assert C.running("p1", tmp_path)
        assert not C.running("p", tmp_path)                                  # another phase's queue (prefix)
    finally:
        q.kill()
        q.wait()
    assert not C.running("p1", tmp_path)


def test_chain_runs_named_phases_only(tmp_path, monkeypatch):
    """REQ-RUN-38: `s2_chain.py s2_ho_cgp` launches s2_ho_cgp alone, behind the same guard; names outside Stage 2 are
    refused."""
    calls = []
    monkeypatch.setattr(C, "running", lambda p, runs: False)
    monkeypatch.setattr(C, "current_code", lambda: C.REGISTERED_CODE)
    (tmp_path / "addendum.txt").write_text("recorded (test)\n")
    monkeypatch.setattr(C, "ADDENDUM", tmp_path / "addendum.txt")                # the CGP addendum recorded

    def run(p, runs, retry=False):
        calls.append(p)
        _phase(runs, p, 1, [dict(key="k", terminal_state="converged")])
    monkeypatch.setattr(C, "run", run)
    assert C.chain(tmp_path, order=("s2_ho_cgp",)) == 0 and calls == ["s2_ho_cgp"]
    assert C.phases_from([]) == C.ORDER and C.phases_from(["s2_ho_cgp"]) == ("s2_ho_cgp",)
    with pytest.raises(SystemExit):
        C.phases_from(["s2_cal_a0"])


def test_s2_ho_cgp_waits_for_the_recorded_addendum(tmp_path, monkeypatch):
    """REQ-RUN-38: s2_ho_cgp launches only once the CGP addendum's OSF link is recorded (D076, D083): the transfer model
    must be registered before any held-out CGP run."""
    calls = []
    monkeypatch.setattr(C, "running", lambda p, runs: False)
    monkeypatch.setattr(C, "current_code", lambda: C.REGISTERED_CODE)
    monkeypatch.setattr(C, "run", lambda p, runs, retry=False: calls.append(p))
    monkeypatch.setattr(C, "ADDENDUM", tmp_path / "stage2_addendum.txt")
    assert C.chain(tmp_path, order=("s2_ho_cgp",)) == 1 and calls == []          # not recorded: refused
    (tmp_path / "stage2_addendum.txt").write_text("filed: https://osf.io/xxxxx\n")
    assert C.chain(tmp_path, order=("s2_ho_cgp",)) == 1 and calls == ["s2_ho_cgp"]

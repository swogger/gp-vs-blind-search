"""Registered task lists (REQ-RUN-19): a confirmatory phase runs exactly the tasks registered with Stage 1."""
import json

import pytest

from reach import s2_lock as L


def _tasks(keys):
    return [dict(key=k, argv=["x"], cpu_limit=1) for k in keys]


def test_hash_ignores_order_and_sees_every_key():
    a = L.task_list_hash(_tasks(["b", "a", "c"]))
    assert a == L.task_list_hash(_tasks(["c", "a", "b"]))
    assert a != L.task_list_hash(_tasks(["a", "b"])) and a != L.task_list_hash(_tasks(["a", "b", "d"]))


def test_unregistered_or_changed_task_lists_are_refused(tmp_path):
    p = tmp_path / "stage1_tasks.json"
    with pytest.raises(PermissionError):                                  # no registered list at all
        L.check_registered_tasks("s2_cal_a0", _tasks(["a"]), p)
    p.write_text(json.dumps({"s2_cal_a0": dict(n_tasks=2, sha256=L.task_list_hash(_tasks(["a", "b"])))}))
    L.check_registered_tasks("s2_cal_a0", _tasks(["b", "a"]), p)          # the registered list passes
    with pytest.raises(PermissionError):
        L.check_registered_tasks("s2_cal_a0", _tasks(["a", "b", "extra"]), p)
    with pytest.raises(PermissionError):
        L.check_registered_tasks("s2_cal_mech", _tasks(["a"]), p)


def test_the_runner_checks_every_confirmatory_phase(tmp_path, monkeypatch):
    """Planning a locked phase through the runner calls the check (planted: an unregistered list must be refused)."""
    import argparse
    from reach.runner import __main__ as M
    monkeypatch.setitem(M.PHASES, "s2_cal_fake", dict(build=lambda: _tasks(["a"]), description="", verify=None))
    monkeypatch.setattr(L, "TASKS_FILE", tmp_path / "none.json")
    with pytest.raises(PermissionError):
        M.cmd_plan(argparse.Namespace(phase="s2_cal_fake", runs=tmp_path, force=False), quiet=True)


def test_the_runner_checks_heldout_phases_against_stage2(tmp_path, monkeypatch):
    import argparse
    from reach.runner import __main__ as M
    monkeypatch.setitem(M.PHASES, "s2_ho_fake", dict(build=lambda: _tasks(["a"]), description="", verify=None))
    monkeypatch.setattr(L, "TASKS2_FILE", tmp_path / "none.json")
    with pytest.raises(PermissionError):
        M.cmd_plan(argparse.Namespace(phase="s2_ho_fake", runs=tmp_path, force=False), quiet=True)


# ------------------------------------------------------- guardrail row 135: the full task list, not its keys only
def _full(keys, **over):
    return [dict(dict(key=k, argv=["{python}", "-m", "reach.tasks.s2_run", "--timeout", "900"], cpu_limit=1000,
                      meta=dict(seed=1)), **over) for k in keys]


def test_full_hash_sees_argv_cpu_limit_env_and_meta_but_not_order_or_weight():
    a = L.task_list_full_hash(_full(["b", "a"]))
    assert a == L.task_list_full_hash(_full(["a", "b"]))
    assert a == L.task_list_full_hash([dict(t, weight=7, cwd=".") for t in _full(["a", "b"])])   # defaults, weight
    for over in (dict(argv=["{python}", "-m", "reach.tasks.s2_run", "--timeout", "1"]), dict(cpu_limit=5),
                 dict(meta=dict(seed=2)), dict(env=dict(X="1")), dict(cwd="legacy")):
        assert L.task_list_full_hash(_full(["a", "b"], **over)) != a, over
    assert L.task_list_hash(_full(["a", "b"], cpu_limit=5)) == L.task_list_hash(_full(["a", "b"]))   # keys only


def test_full_hash_is_the_same_in_every_worktree(monkeypatch, tmp_path):
    """Operon tasks carry absolute paths of the checkout (.venv-ext); the hash replaces the repository root, so the
    dev and main worktrees (and CI) give the same value."""
    def tasks(root):
        return [dict(key="a", argv=[f"{root}/.venv-ext/bin/python", "-m", "reach.tasks.s2_operon"], cpu_limit=600,
                     env=dict(DYLD_LIBRARY_PATH=f"{root}/.venv-ext/operon-libs"))]
    monkeypatch.setattr(L, "_ROOT", tmp_path / "one")
    h1 = L.task_list_full_hash(tasks(tmp_path / "one"))
    monkeypatch.setattr(L, "_ROOT", tmp_path / "two")
    assert L.task_list_full_hash(tasks(tmp_path / "two")) == h1
    assert L.task_list_full_hash(tasks(tmp_path / "one")) != h1             # another root's path is not normalised


def test_a_changed_setting_with_the_same_keys_is_refused(tmp_path):
    """Audit 2026-10-28, finding 2: every s2_ho_a0 task planted with `--timeout 1` and cpu_limit 5 passed the key-only
    check. A registered list with its full hash refuses it; a Stage 1 entry without one keeps the key check."""
    p = tmp_path / "stage2_tasks.json"
    reg = _full(["a", "b"])
    p.write_text(json.dumps({"s2_ho_a0": dict(n_tasks=2, sha256=L.task_list_hash(reg),
                                              sha256_full=L.task_list_full_hash(reg))}))
    L.check_registered_tasks("s2_ho_a0", _full(["b", "a"]), p)                # the registered list passes
    for over in (dict(argv=["{python}", "-m", "reach.tasks.s2_run", "--timeout", "1"]), dict(cpu_limit=5)):
        with pytest.raises(PermissionError, match="settings"):
            L.check_registered_tasks("s2_ho_a0", _full(["a", "b"], **over), p)
    p.write_text(json.dumps({"s2_cal_a0": dict(n_tasks=2, sha256=L.task_list_hash(reg))}))
    L.check_registered_tasks("s2_cal_a0", _full(["a", "b"], cpu_limit=5), p)  # Stage 1: keys only, as registered


def test_stage2_task_lists_register_the_full_hash():
    """Every phase in stage2_tasks.json carries `sha256_full` (scripts/s2_task_lists.py --stage 2)."""
    if not L.TASKS2_FILE.exists():
        pytest.skip("no Stage 2 task lists yet")
    reg = json.loads(L.TASKS2_FILE.read_text())
    assert set(reg) == set(L.STAGE2_PHASES)
    assert all(len(v.get("sha256_full", "")) == 64 for v in reg.values())


def test_every_stage2_phase_builds_its_registered_full_list(tmp_path, monkeypatch, request):
    """Each Stage 2 phase, built as registered (locks lifted in memory only), passes the full check; the same build with
    a planted soft CPU limit of 1 s for E0 runs (keys unchanged) is refused."""
    from reach.runner import phases as P
    if not L.TASKS2_FILE.exists():
        pytest.skip("no Stage 2 task lists yet")
    request.getfixturevalue("cgp_exe")                    # CGP phases need the binary to plan (REQ-RUN-27)
    lock = tmp_path / "stage.txt"
    lock.write_text("test build only\n")
    monkeypatch.setattr(L, "STAGE1_FILE", lock)
    monkeypatch.setattr(L, "STAGE2_FILE", lock)
    for name in L.STAGE2_PHASES:
        L.check_registered_tasks(name, P.registration_build(name)(), L.TASKS2_FILE)
    monkeypatch.setattr(P, "S2_TIMEOUT", {**P.S2_TIMEOUT, 1: 1})
    planted = P.registration_build("s2_ho_a0")()
    assert L.task_list_hash(planted) == json.loads(L.TASKS2_FILE.read_text())["s2_ho_a0"]["sha256"]   # same keys
    with pytest.raises(PermissionError, match="settings"):
        L.check_registered_tasks("s2_ho_a0", planted, L.TASKS2_FILE)

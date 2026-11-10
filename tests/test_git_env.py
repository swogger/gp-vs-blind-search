"""Guardrail row 178: tests never inherit a hook's git environment (tests/conftest.py strips it)."""
import os
import subprocess

from conftest import GIT_ENV, strip_git_env


def _git(args, cwd, env):
    return subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", *args], cwd=cwd, env=env,
                          check=True, capture_output=True, text=True).stdout.strip()


def test_tests_never_inherit_a_hooks_git_environment(tmp_path):
    assert not [k for k in GIT_ENV if k in os.environ]
    base = strip_git_env(dict(os.environ))
    decoy, wt = tmp_path / "decoy", tmp_path / "wt"
    decoy.mkdir()
    _git(["init", "-q"], decoy, base)
    _git(["commit", "-q", "--allow-empty", "-m", "init"], decoy, base)
    _git(["worktree", "add", "-q", str(wt), "-b", "wt"], decoy, base)
    gitdir = decoy / ".git/worktrees/wt"
    hook = dict(base, GIT_DIR=str(gitdir), GIT_INDEX_FILE=str(gitdir / "index"))   # what a linked worktree's hook exports
    # plant: a planted repository's git run under the hook's environment damages the decoy (the check fires)
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "f").write_text("x")
    _git(["init", "-q"], planted, hook)
    _git(["add", "f"], planted, hook)
    _git(["commit", "-qm", "planted"], planted, hook)
    assert _git(["config", "core.bare"], decoy, base) == "true"
    assert _git(["log", "-1", "--format=%s", "wt"], decoy, base) == "planted"
    # with the environment stripped (as conftest does) the planted repository is its own and the decoy is untouched
    _git(["config", "core.bare", "false"], decoy, base)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "f").write_text("y")
    env = strip_git_env(dict(hook))
    _git(["init", "-q"], clean, env)
    _git(["add", "f"], clean, env)
    _git(["commit", "-qm", "own"], clean, env)
    assert (clean / ".git").is_dir() and _git(["config", "core.bare"], decoy, base) == "false"
    assert _git(["log", "-1", "--format=%s", "wt"], decoy, base) == "planted"
    assert _git(["log", "-1", "--format=%s"], clean, base) == "own"

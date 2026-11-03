"""Guardrails written before the Study 2 code (D049; docs/requirements/guardrails.md).

Each test guards a past error or a rule. `test_known_defect_*` tests document defects found in the review of
2026-10-19; they are strict xfails, so they fail loudly once the defect is fixed and must then become normal tests
(and the requirement's status becomes active).
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from reach import grammar as G
from reach.engines.treegp import run_gp

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMPORT_COMMIT = "4d3cf271776e50ddb33329aeb2e598dae638d4ce"     # D047: legacy/, docs/reference/, reference_code/
READ_ONLY = ("legacy", "docs/reference", "reference_code")
DOC_MAX_LINES = 200
DOC_EXEMPT = {"decisions.md"}                                   # append-only log, read by heading


# ------------------------------------------------------------------ requirements and docs
def test_requirements_traceable():
    spec = importlib.util.spec_from_file_location("check_requirements", ROOT / "scripts/check_requirements.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    problems, counts = mod.check()
    assert sum(counts.values()) > 0, "no requirements found"
    assert problems == []


def _doc_files():
    return sorted(p for p in DOCS.rglob("*.md") if "reference" not in p.relative_to(DOCS).parts)


def test_docs_index_complete():
    index = (DOCS / "README.md").read_text()
    links = set(re.findall(r"\]\(([^)#\s]+)\)", index))
    for link in links:
        if not link.startswith("http"):
            assert (DOCS / link).exists(), f"docs/README.md links to a missing path: {link}"
    for p in _doc_files():
        rel = str(p.relative_to(DOCS))
        if rel != "README.md":
            assert rel in links, f"docs/{rel} is not listed in docs/README.md"


def test_docs_files_are_small():
    for p in _doc_files():
        if p.name in DOC_EXEMPT:
            continue
        n = len(p.read_text().splitlines())
        assert n <= DOC_MAX_LINES, f"{p.relative_to(ROOT)} has {n} lines (max {DOC_MAX_LINES}); split it"


# ------------------------------------------------------------------ code patterns
def _source_files():
    return [p for p in (ROOT / "src/reach").rglob("*.py") if "galib" not in p.parts]


def test_no_global_rng_in_new_code():
    """Lesson 5: ga-lib's global `random` is patched per run; every other module uses explicit generators."""
    bad = re.compile(r"\brandom\.(choice|random|randint|sample|uniform|shuffle|gauss|seed)\(|"
                     r"\bnp\.random\.(seed|rand|randn|randint|choice|shuffle|permutation)\(")
    hits = [f"{p.relative_to(ROOT)}:{n}" for p in _source_files()
            for n, line in enumerate(p.read_text().splitlines(), 1) if bad.search(line)]
    assert hits == []


def test_no_size_penalty_in_source():
    """Lesson 1: no size term in any fitness. The only mention allowed is the flag that switches it off in the
    legacy scripts."""
    hits = [f"{p.relative_to(ROOT)}:{n}" for p in _source_files()
            for n, line in enumerate(p.read_text().splitlines(), 1)
            if "size_penalty" in line and "--no_size_penalty" not in line]
    assert hits == []


WALL_CLOCK = re.compile(r"\btime\.time\(\)|\btime\.monotonic\(\)|\btime\.perf_counter\(\)|datetime\.now|timeout_in_seconds")
LIMIT_WORDS = re.compile(r"timeout|budget|limit|deadline|max_time|cutoff", re.I)
WALL_LIMIT_ALLOWED = {"src/reach/tasks/pysr_check.py"}     # paper 1 only (PySR's own wall timeout); not used in Study 2


def test_limits_use_cpu_time_not_wall_clock():
    """Rule (author, 2026-10-19; lesson 3): every run limit and budget is CPU time. Wall clock may only be read for
    progress, ETAs and the recorded wall_s, never on a line that compares against a limit or budget."""
    hits = []
    for p in _source_files():
        rel = str(p.relative_to(ROOT))
        if rel in WALL_LIMIT_ALLOWED:
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if WALL_CLOCK.search(line) and LIMIT_WORDS.search(line):
                hits.append(f"{rel}:{n}: {line.strip()}")
    assert hits == []


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs the git history")
def test_read_only_trees_unchanged():
    """Rule 13: legacy/, docs/reference/ and reference_code/ are never edited after their import (D047)."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *READ_ONLY], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    assert dirty == "", f"uncommitted changes in read-only trees:\n{dirty}"
    commits = subprocess.run(["git", "rev-list", "HEAD", "--", *READ_ONLY], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    if commits and IMPORT_COMMIT not in commits:
        pytest.skip("shallow history: the import commit is not available")
    assert commits == [IMPORT_COMMIT], f"read-only trees changed after import: {commits}"


# ------------------------------------------------------------------ engine invariants
def _target(grammar, path, target_id):
    t = next(t for t in json.loads((ROOT / path).read_text()) if t["target_id"] == target_id)
    g = G.get(grammar)
    tree = G.parse_expr(t["expression"], g)
    kw = {}
    if g.kind == "real":
        grid, val = G.sample_grid(g), G.sample_val(g, 20260721 + 1)
        kw = dict(val_xs=val, val_target=G.evaluate(g, tree, val), var_norm=t["var_target"], grid_xs=grid,
                  grid_target=G.evaluate(g, tree, grid), grid_scale=t["target_scale"], undefined_penalty="scaled")
    return g, tree, kw


@pytest.mark.parametrize("grammar,path,targets,seeds", [
    ("BOOL-AON-n5", "data/targets/BOOL-AON-n5/targets.json", ["BOOL-AON-n5_K15_03", "BOOL-AON-n5_K12_00"], [12, 11]),
    ("EML", "data/targets/EML/targets_v2.json", ["EML_K09_03", "EML_K09_05"], [11, 12]),
])
def test_prefix_property_gp(grammar, path, targets, seeds):
    """C2 (D048): a run of L generations is exactly the start of a longer run with the same seed, so one long run
    gives the outcome at every shorter budget. Checked at the generation of the first exact match."""
    for tid in targets:
        g, tree, kw = _target(grammar, path, tid)
        for seed in seeds:
            full = run_gp(g, tree, "A", seed, generations=200, stop_on="exact", **kw)
            if not full["converged"]:
                continue
            k = full["gens_run"]
            short = run_gp(g, tree, "A", seed, generations=k, stop_on="exact", **kw)
            just = run_gp(g, tree, "A", seed, generations=k + 1, stop_on="exact", **kw)
            assert not short["converged"]
            assert just["converged"] and just["gens_run"] == k
            assert (just["best_expr"], just["final_nmse"]) == (full["best_expr"], full["final_nmse"])
            return
    pytest.fail(f"no converging (target, seed) pair for {grammar}: the precondition of this test changed")


# ------------------------------------------------------------------ known defects (review 2026-10-19)
@pytest.mark.xfail(strict=True, raises=TypeError, reason="F4: run_es needs real-domain data arguments")
def test_known_defect_es_boolean():
    from reach.engines.es import run_es
    g = G.get("BOOL-NAND-n3")
    r = run_es(g, G.parse_expr("nand(x1, x2)", g), 1, evaluations=400)
    assert r["converged"]


@pytest.mark.xfail(strict=True, reason="F5: random sampling reads target_scale, which Boolean targets lack")
def test_known_defect_random_sample_boolean():
    p = subprocess.run([sys.executable, "-m", "reach.tasks.random_sample", "--grammar", "BOOL-AON-n5",
                        "--targets", "data/targets/BOOL-AON-n5/targets.json", "--n", "100", "--seed", "1"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-300:]


def test_registered_task_lists_still_build(tmp_path, monkeypatch):
    """Guardrail row 61: every confirmatory phase must still build exactly the task list registered with Stage 1
    (data/registration/stage1_tasks.json). A code change that alters a list would make the runner refuse a queued phase
    in the middle of calibration. The Stage 1 lock is lifted in memory only; nothing runs."""
    from reach import s2_lock as L
    from reach.runner.phases import PHASES
    if not L.TASKS_FILE.exists():
        pytest.skip("no registered task lists")
    lock = tmp_path / "stage1.txt"
    lock.write_text("test build only\n")
    monkeypatch.setattr(L, "STAGE1_FILE", lock)
    reg = json.loads(L.TASKS_FILE.read_text())
    for name, want in reg.items():
        tasks = PHASES[name]["build"]()
        assert (len(tasks), L.task_list_hash(tasks)) == (want["n_tasks"], want["sha256"]), name


def test_registered_stage2_task_lists_still_build(tmp_path, monkeypatch, request):
    """Guardrail rows 61 and 89 for Stage 2: every phase registered in data/registration/stage2_tasks.json (CGP
    calibration and the held-out phases) still builds exactly its registered list. The locks are lifted in memory only;
    nothing runs. s2_ho_cgp is built as registered, without the H10b transfer model (phases.registration_build)."""
    from reach import s2_lock as L
    from reach.runner.phases import registration_build
    if not L.TASKS2_FILE.exists():
        pytest.skip("no Stage 2 task lists registered yet")
    request.getfixturevalue("cgp_exe")                    # CGP phases need the binary to plan (REQ-RUN-27)
    lock = tmp_path / "stage.txt"
    lock.write_text("test build only\n")
    monkeypatch.setattr(L, "STAGE1_FILE", lock)
    monkeypatch.setattr(L, "STAGE2_FILE", lock)
    for name, want in json.loads(L.TASKS2_FILE.read_text()).items():
        assert name in L.STAGE2_PHASES, name
        tasks = registration_build(name)()
        assert (len(tasks), L.task_list_hash(tasks)) == (want["n_tasks"], want["sha256"]), name


@pytest.mark.study_machine
def test_running_confirmatory_phases_can_resume():
    """Guardrail row 62: a confirmatory phase with results must have exactly one code hash and that snapshot on disk,
    so it resumes on the same code after a reboot (row 54). Reads only the code_hash field of each row. The snapshots are
    untracked and live where the queue runs, the main worktree (row 121), so a dev worktree checks them there."""
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip())
    queue_root = (common if common.is_absolute() else ROOT / common).resolve().parent
    for res in sorted((ROOT / "data/runs").glob("s2_cal_*/results.jsonl")):
        hashes = set()
        for line in res.read_text().splitlines():
            if line.strip():
                try:
                    hashes.add(json.loads(line).get("code_hash"))
                except json.JSONDecodeError:
                    pass                                          # a truncated last line is ignored by the queue too
        hashes.discard(None)
        assert len(hashes) <= 1, f"{res.parent.name}: mixed code versions {hashes}"
        for h in hashes:
            snap = queue_root / "data/runs" / res.parent.name / "code" / h / "reach"
            assert snap.exists(), f"{res.parent.name}: snapshot {h} missing in {queue_root}"


def test_watchdog_agent_keeps_the_queues_it_starts_and_runs_at_normal_priority():
    """Guardrail rows 71 and 72: cron throttled workers to background priority; the launch agent killed the queues it
    started when its script exited. Both settings must stay in the agent definition."""
    import plistlib
    p = plistlib.loads((ROOT / "scripts/launchd/com.gpsearch.keeprunning.plist").read_bytes())
    assert p["ProcessType"] == "Interactive" and p["AbandonProcessGroup"] is True
    assert p["StartInterval"] == 900 and p["RunAtLoad"] is True


def test_ci_wait_reports_the_commit_it_watched():
    """Guardrail row 120: ci_wait.sh watched the run of the commit pushed (HEAD at start) but printed the subject of HEAD
    at the end, so a commit made while CI ran was reported as passed although it was never pushed. Every commit it names
    must be the watched one."""
    import re
    text = (ROOT / "scripts/ci_wait.sh").read_text()
    logs = re.findall(r"git log --oneline -1[^)\n]*", text)
    assert logs and all('"$sha"' in x for x in logs), logs

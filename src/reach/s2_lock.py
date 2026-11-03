"""Held-out lock (D048 C1; REQ-RUN-10). Kept apart from `s2_sets.py`, whose bytes seed the held-out draw and must not
change after it (test_set_list_frozen_after_draw)."""
import json as _json

from reach.paths import ROOT as _ROOT

DRAW_FILE = _ROOT / "data/registration/heldout_draw.json"          # written by reach.tasks.s2_draw
STAGE2_FILE = _ROOT / "data/registration/stage2_registration.txt"  # the Stage 2 OSF link, recorded by Claude


RANDOM_FILE = _ROOT / "data/registration/random_draw.json"      # written by reach.tasks.s2_random_draw (T26)


def held_out():
    """Listed sets drawn as held out, plus every drawn random-set candidate (empty before the draws)."""
    out = set()
    if DRAW_FILE.exists():
        rec = _json.loads(DRAW_FILE.read_text())
        out |= {g for d in ("real", "bool") for g in rec[d]["held_out"]}
    if RANDOM_FILE.exists():
        rec = _json.loads(RANDOM_FILE.read_text())
        out |= {c["name"] for d in ("real", "bool") for c in rec[d]["candidates"]}
    return out


def signature(name):
    """What a set is, whatever its name: domain, sorted operators, constants, inputs (review finding 6)."""
    from reach import grammar as G
    g = G.get(name)
    return (g.kind, tuple(sorted(g.operators)), tuple(sorted(g.constants)), getattr(g, "n_inputs", 1))


_LOCKED = {}


def _locked_signatures():
    key = tuple((str(p), p.stat().st_mtime) if p.exists() else (str(p), None) for p in (DRAW_FILE, RANDOM_FILE))
    if key not in _LOCKED:
        _LOCKED.clear()
        _LOCKED[key] = {signature(n) for n in held_out()}
    return _LOCKED[key]


def check_search_allowed(grammar):
    """Every Study 2 search phase calls this for each task: held-out sets, and any set with the same operators,
    constants and inputs under another name, stay locked until Stage 2 is filed."""
    if STAGE2_FILE.exists() and STAGE2_FILE.read_text().strip():
        return
    if grammar in held_out() or signature(grammar) in _locked_signatures():
        raise PermissionError(f"{grammar} is a held-out set: no search runs before the Stage 2 registration "
                              f"is recorded in {STAGE2_FILE}")


STAGE1_FILE = _ROOT / "data/registration/stage1_registration.txt"  # the Stage 1 OSF link, recorded by Claude


def check_stage3():
    """The H6d re-run phases (D098) plan only once the Stage 3 registration is recorded."""
    if not (STAGE3_FILE.exists() and STAGE3_FILE.read_text().strip()):
        raise PermissionError(f"H6d re-runs: the Stage 3 registration is not recorded in {STAGE3_FILE}")


def check_stage1():
    """Every confirmatory phase calls this when it is planned (REQ-RUN-18): no confirmatory run before Stage 1."""
    if not (STAGE1_FILE.exists() and STAGE1_FILE.read_text().strip()):
        raise PermissionError(f"confirmatory phase: the Stage 1 registration is not recorded in {STAGE1_FILE}")


TASKS_FILE = _ROOT / "data/registration/stage1_tasks.json"   # task-list hashes attached to Stage 1 (REQ-RUN-19)
TASKS2_FILE = _ROOT / "data/registration/stage2_tasks.json"  # the same for the held-out phases (Stage 2)
STAGE3_FILE = _ROOT / "data/registration/stage3_registration.txt"  # the Stage 3 OSF link (H6d; D098)
TASKS3_FILE = _ROOT / "data/registration/stage3_tasks.json"  # the H6d re-run phases (Stage 3)
STAGE3_PHASES = ("s2_revisit_a0", "s2_revisit_a16")


# Phases whose task lists are registered with Stage 2 (stage2_tasks.json): the held-out phases and the CGP calibration,
# which is registered in Stage 2 although its name starts with s2_cal_ (D076; guardrail row 89).
STAGE2_PHASES = ("s2_cal_cgp", "s2_ho_a0", "s2_ho_a16", "s2_ho_bench", "s2_ho_operon", "s2_ho_cgp")


STAGE4_FILE = _ROOT / "data/registration/stage4_registration.txt"  # the Stage 4 OSF link (H11; D100)
TASKS4_FILE = _ROOT / "data/registration/stage4_tasks.json"  # the Stage 4 restart phase
STAGE4_PHASES = ("s4_restart",)


def check_stage4():
    """The Stage 4 phases (remedy test, D100) plan only once the Stage 4 registration is recorded."""
    if not (STAGE4_FILE.exists() and STAGE4_FILE.read_text().strip()):
        raise PermissionError(f"Stage 4 phase: the Stage 4 registration is not recorded in {STAGE4_FILE}")


def registered_tasks_file(phase):
    """The registered task-list file a phase must match when planned: Stage 2 phases (and any other s2_ho_ search
    phase) -> TASKS2_FILE; the other s2_cal_ phases -> TASKS_FILE (Stage 1); anything else -> None (not checked)."""
    if phase in STAGE2_PHASES or (phase.startswith("s2_ho_") and phase != "s2_ho_targets"):
        return TASKS2_FILE
    if phase in STAGE3_PHASES:
        return TASKS3_FILE
    if phase in STAGE4_PHASES:
        return TASKS4_FILE
    if phase.startswith("s2_cal_"):
        return TASKS_FILE
    return None


def task_list_hash(tasks):
    """SHA-256 of the sorted task keys. A key fixes set, target, algorithm, configuration, run length and seed."""
    import hashlib
    return hashlib.sha256("\n".join(sorted(t["key"] for t in tasks)).encode()).hexdigest()


FULL_FIELDS = ("key", "argv", "cwd", "cpu_limit", "env", "meta")   # everything a task is run with; weight only orders


def _rel(x):
    """Strings under this checkout's root are written relative to it ("{root}/..."), so every worktree agrees."""
    if isinstance(x, str):
        root = str(_ROOT)
        return "{root}" + x[len(root):] if x == root or x.startswith(root + "/") else x
    if isinstance(x, list):
        return [_rel(v) for v in x]
    if isinstance(x, dict):
        return {k: _rel(v) for k, v in x.items()}
    return x


def task_list_full_hash(tasks):
    """SHA-256 of the whole task list (guardrail row 135): per task, sorted by key, its key, argv, cwd, CPU limit, env
    and meta with the runner's defaults (cwd ".", env and meta empty) and the repository root written as "{root}".
    Registered beside the key hash for Stage 2, so a changed setting under the same keys (a timeout, a CPU limit, a
    target checksum) is refused when the phase is planned."""
    import hashlib
    rows = []
    for t in sorted(tasks, key=lambda t: t["key"]):
        d = dict(cwd=".", env={}, meta={})
        d.update({k: t[k] for k in FULL_FIELDS if t.get(k) is not None})
        rows.append({k: _rel(d.get(k)) for k in FULL_FIELDS})
    return hashlib.sha256(_json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_registered_tasks(name, tasks, path=None):
    """A confirmatory phase runs exactly the task list registered with its stage (checked when the runner plans it):
    the same keys (count and key hash), and, where the registration records it (Stage 2), the same full task list."""
    p = path or TASKS_FILE
    reg = _json.loads(p.read_text()) if p.exists() else {}
    if name not in reg:
        raise PermissionError(f"{name}: no registered task list in {p}")
    if reg[name]["sha256"] != task_list_hash(tasks) or reg[name]["n_tasks"] != len(tasks):
        raise PermissionError(f"{name}: the task list differs from the one registered in {p}")
    if "sha256_full" in reg[name] and reg[name]["sha256_full"] != task_list_full_hash(tasks):
        raise PermissionError(f"{name}: the same task keys, but their settings (argv, CPU limit, env or meta) differ "
                              f"from the ones registered in {p}")


def check_stage2():
    """Every held-out phase calls this when it is planned: no held-out run before the Stage 2 registration."""
    if not (STAGE2_FILE.exists() and STAGE2_FILE.read_text().strip()):
        raise PermissionError(f"held-out phase: the Stage 2 registration is not recorded in {STAGE2_FILE}")

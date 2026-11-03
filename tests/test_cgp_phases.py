"""CGP validation and pilot phases (D072 step 3; docs/design/cgp.md), written before the code: REQ-RUN-25 to 27."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reach import cgp
from reach import grammar as G
from reach.runner import phases as P
from reach.seeds import s2_seeds

ROOT = Path(__file__).resolve().parents[1]
PILOT_SETS = ("F-EML", "R-ELEM4", "BOOL-NAND-n5", "BOOL-AONX-n5")


def _arg(argv, flag):
    return argv[argv.index(flag) + 1]


def _settings(task):
    a = task["argv"]
    return dict(set=_arg(a, "--set"), n=int(_arg(a, "--n_nodes")), budget=float(_arg(a, "--budget")),
                seed=int(_arg(a, "--seed")), levels=_arg(a, "--levels_back"), mutation=_arg(a, "--mutation"),
                strategy=_arg(a, "--strategy"), stop=_arg(a, "--stop_on"), penalty=_arg(a, "--penalty"),
                timeout=float(_arg(a, "--timeout")), targets=_arg(a, "--targets"))


# ------------------------------------------------------------------ REQ-RUN-25: validation
def test_validation_target_is_even_3_parity():
    """Goldman and Punch's 3-bit parity is even parity: output 1 when the number of ones is even."""
    p = ROOT / "data/targets_s2/BOOL-AONN-n3/validation.json"
    sha = (ROOT / "data/targets_s2/BOOL-AONN-n3/validation.sha256").read_text().split()[0]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == sha
    (t,) = json.loads(p.read_text())
    g = G.get(cgp.VALIDATION_SET)
    assert list(g.operators) == ["and", "or", "nand", "nor"] and list(g.variables) == ["x1", "x2", "x3"]
    assert not g.constants
    vm = G.var_masks(3)
    even = sum(1 << r for r in range(8) if sum((vm[v] >> r) & 1 for v in g.variables) % 2 == 0)
    assert G.evaluate(g, G.parse_expr(t["expression"], g)) == even


def test_validation_phase_is_goldman_punch_table_ii(cgp_exe):
    tasks = P.PHASES["s2_cgp_validate"]["build"]()
    assert len(tasks) == len({t["key"] for t in tasks}) == 2 * 200
    seeds = s2_seeds("cgp_validation")
    assert len(seeds) == 200
    by_n = {}
    for t in tasks:
        s = _settings(t)
        assert s["set"] == "BOOL-AONN-n3" and s["targets"] == "data/targets_s2/BOOL-AONN-n3/validation.json"
        assert (s["levels"], s["mutation"], s["strategy"], s["stop"]) == ("unrestricted", "single", "1+4", "exact")
        assert round(s["budget"] * cgp.E0) == 10 ** 7                    # up to 10^7 evaluations
        assert t["cpu_limit"] > s["timeout"] > 0                          # CPU limits: soft in the engine, hard above
        assert t["argv"][:3] == ["{python}", "-m", "reach.tasks.s2_cgp"]  # one run per task through the CGP task
        by_n.setdefault(s["n"], []).append(s["seed"])
    assert set(by_n) == {500, 3000}
    for n, ss in by_n.items():
        assert sorted(ss) == sorted(seeds)


def test_phase_tasks_run_through_the_cgp_task(cgp_exe):
    """One validation task and one pilot task, exactly as the queue would run them, produce a CGP record."""
    val = P.PHASES["s2_cgp_validate"]["build"]()
    pil = [t for t in P.PHASES["s2_pilot_cgp"]["build"]() if t["meta"]["set"] == "BOOL-NAND-n5"]
    pil = min(pil, key=lambda t: (t["meta"]["K"], t["meta"]["n_nodes"]))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    for task, n_nodes, budget_e in ((val[0], val[0]["meta"]["n_nodes"], 10 ** 7), (pil, pil["meta"]["n_nodes"], 320_000)):
        argv = [sys.executable if a == "{python}" else a for a in task["argv"]]
        p = subprocess.run(argv, cwd=ROOT / task["cwd"] if "cwd" in task else ROOT, capture_output=True, text=True,
                           env=env, timeout=300)
        assert p.returncode == 0, p.stderr[-800:]
        r = json.loads(next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))[7:])
        assert r["algorithm"] == "CGP" and r["n_nodes"] == n_nodes and r["budget_E"] == budget_e
        assert r["exact"] and r["eval_index_exact"] >= 1 and r["terminal_state"] == "converged"


# ------------------------------------------------------------------ REQ-RUN-26: pilot
def test_pilot_phase_runs_every_pilot_target_at_three_lengths(cgp_exe):
    tasks = P.PHASES["s2_pilot_cgp"]["build"]()
    targets = {g: json.loads((ROOT / f"data/targets_s2/{g}/pilot.json").read_text()) for g in PILOT_SETS}
    assert len(tasks) == len({t["key"] for t in tasks}) == sum(len(v) for v in targets.values()) * 3 * 3
    seeds = s2_seeds("cgp_pilot")[:3]
    assert not set(seeds) & set(s2_seeds("pilot")) and not set(seeds) & set(s2_seeds("calibration"))
    cells = set()
    for t in tasks:
        s = _settings(t)
        assert s["targets"] == f"data/targets_s2/{s['set']}/pilot.json"
        assert s["n"] in (100, 500, 2000) and s["seed"] in seeds and s["budget"] == 16
        assert (s["levels"], s["mutation"], s["strategy"], s["stop"], s["penalty"]) == \
            ("unrestricted", "single", "1+4", "exact", "scaled")
        assert t["cpu_limit"] > s["timeout"] >= 600
        cells.add((s["set"], _arg(t["argv"], "--target_id"), s["n"], s["seed"]))
    assert {(g, t["target_id"], n, s) for g in PILOT_SETS for t in targets[g] for n in (100, 500, 2000)
            for s in seeds} == cells


def test_cgp_tasks_refuse_calibration_and_held_out_targets(cgp_exe, tmp_path, monkeypatch):
    from reach import s2_lock
    monkeypatch.setattr(s2_lock, "STAGE2_FILE", tmp_path / "not_recorded.txt")   # before Stage 2, whatever the repo holds
    t = json.loads((ROOT / "data/targets_s2/BOOL-NAND-n5/pilot.json").read_text())[0]
    assert P._cgp_task("x", "BOOL-NAND-n5", "pilot", t, 100, 1, 16, 600)          # control: pilot targets pass
    for tname in ("calibration", "pool", "heldout", "benchmark"):
        with pytest.raises(PermissionError, match="Stage 2"):
            P._cgp_task("x", "BOOL-NAND-n5", tname, t, 100, 1, 16, 600)
    from reach.s2_lock import held_out, STAGE2_FILE
    if held_out() and not (STAGE2_FILE.exists() and STAGE2_FILE.read_text().strip()):
        with pytest.raises(PermissionError):
            P._cgp_task("x", sorted(held_out())[0], "pilot", t, 100, 1, 16, 600)


# ------------------------------------------------------------------ REQ-RUN-27: binary present before planning
def test_planning_without_the_binary_fails_clearly(tmp_path, monkeypatch, cgp_exe):
    from reach.runner.__main__ import main
    runs = tmp_path / "runs"
    monkeypatch.setattr(cgp, "STORE", tmp_path / "empty_store")
    for ph in ("s2_cgp_validate", "s2_pilot_cgp"):
        with pytest.raises(cgp.BinaryError, match=r"cgp\.build\(\)"):
            main(["--runs", str(runs), "plan", ph])
        assert not (runs / ph / "manifest.json").exists()
    monkeypatch.undo()                                                    # control: with the binary it plans
    main(["--runs", str(runs), "plan", "s2_cgp_validate"])
    assert json.loads((runs / "s2_cgp_validate/manifest.json").read_text())["n_tasks"] == 400

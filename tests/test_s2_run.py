"""T6 run task (written before the code). REQ-RUN-08, REQ-RUN-10, REQ-RUN-15, REQ-MEA-10, REQ-ENG-08."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"set", "domain", "algorithm", "config", "seed", "target_id", "K_proven", "K_construction", "units",
          "budget_E", "generations", "pop_size", "evals_per_gen", "penalty", "stop_on", "seed_leaves", "exact",
          "converged", "gen_exact", "gen_tol", "gens_reached", "terminal_state", "final_error", "final_tree_size",
          "cpu_seconds", "wall_seconds", "best_expr", "targets_sha256"}


def _targets(tmp_path, name, expr, **extra):
    t = [dict(target_id="t0", expression=expr, K_construction=3, K_proven=3, units=dict(nodes=3.0), **extra)]
    p = tmp_path / "targets.json"
    p.write_text(json.dumps(t))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _run(args):
    return subprocess.run([sys.executable, "-m", "reach.tasks.s2_run", *args], cwd=ROOT, capture_output=True, text=True)


def _args(path, sha, s="BOOL-NAND-n3", **over):
    a = dict(set=s, targets=str(path), target_id="t0", algorithm="A", config="base", seed="7", generations="50",
             pop_size="100", penalty="scaled", stop_on="exact", seed_leaves="off", timeout="60", targets_sha256=sha)
    a.update(over)
    return [x for k, v in a.items() if v is not None for x in (f"--{k}", v)]


def _row(p):
    line = next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


def test_boolean_run_records_the_schema(tmp_path):
    path, sha = _targets(tmp_path, "BOOL-NAND-n3", "nand(x1, x2)")
    p = _run(_args(path, sha))
    assert p.returncode == 0, p.stderr[-500:]
    r = _row(p)
    assert FIELDS <= set(r)
    assert r["exact"] and r["terminal_state"] == "converged"
    assert r["gen_exact"] == r["gens_reached"] and r["budget_E"] == 100 * 50 and r["evals_per_gen"] == 100


def test_every_scientific_setting_is_required(tmp_path):
    """REQ-RUN-08 (C6): no defaults for scientific settings; a missing one is an error."""
    path, sha = _targets(tmp_path, "BOOL-NAND-n3", "nand(x1, x2)")
    for missing in ("algorithm", "config", "penalty", "stop_on", "seed_leaves", "generations", "pop_size", "timeout",
                    "targets_sha256"):
        p = _run(_args(path, sha, **{missing: None}))
        assert p.returncode != 0 and f"--{missing}" in p.stderr, missing


def test_changed_target_file_is_refused(tmp_path):
    """REQ-MEA-10: a target file whose checksum differs from the registered one is refused."""
    path, sha = _targets(tmp_path, "BOOL-NAND-n3", "nand(x1, x2)")
    p = _run(_args(path, "0" * 64))
    assert p.returncode != 0 and "checksum" in p.stderr


def test_held_out_set_is_refused(tmp_path):
    """REQ-RUN-10: the real draw holds out R-ARITH; a search run on it is refused before Stage 2."""
    from reach.s2_lock import STAGE2_FILE, held_out
    if "R-ARITH" not in held_out() or STAGE2_FILE.exists():
        pytest.skip("draw not present or Stage 2 already filed")
    path, sha = _targets(tmp_path, "R-ARITH", "+(x, 1)", var_target=0.08, target_scale=4.0)
    p = _run(_args(path, sha, s="R-ARITH"))
    assert p.returncode != 0 and "held-out" in p.stderr


def test_cpu_limit_records_the_generation_reached(tmp_path):
    """REQ-RUN-15: a run stopped by its CPU limit records the generation it reached."""
    path, sha = _targets(tmp_path, "BOOL-NAND-n3", "nand(nand(x1, nand(x2, x3)), nand(x3, nand(x1, x2)))")
    r = _row(_run(_args(path, sha, timeout="0", generations="1000")))
    assert r["terminal_state"] == "timeout" and r["gens_reached"] < 1000 and r["gen_exact"] is None


def test_real_run_uses_the_explicit_penalty(tmp_path):
    from reach import grammar as G
    import numpy as np
    g = G.get("F-EML")
    v = G.evaluate(g, G.parse_expr("eml(x, 1)", g), G.grid_sample())
    path, sha = _targets(tmp_path, "F-EML", "eml(x, 1)", var_target=float(np.mean(np.abs(v - v.mean()) ** 2)),
                         target_scale=float(np.abs(v).max()))
    r = _row(_run(_args(path, sha, s="F-EML", penalty="scaled")))
    assert r["penalty"] == "scaled" and r["domain"] == "real" and r["exact"]


def test_configuration_population_must_match(tmp_path):
    """T19: a phase cannot run configuration R2 (population 1,000) with population 100 by mistake."""
    path, sha = _targets(tmp_path, "BOOL-NAND-n3", "nand(x1, x2)")
    p = _run(_args(path, sha, config="R2"))
    assert p.returncode != 0 and "population" in p.stderr
    r = _row(_run(_args(path, sha, config="R4")))                     # lexicase runs through the task
    assert r["config"] == "R4" and r["exact"]

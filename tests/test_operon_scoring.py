"""Operon scoring per D056 (T20), written before the code; the end-to-end Operon run needs .venv-ext (skipped otherwise)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reach import grammar as G
from reach.tasks import s2_operon_score as S

ROOT = Path(__file__).resolve().parents[1]


def test_parse_and_evaluate_operon_strings():
    x = G.grid_sample().real
    v = S.evaluate_model("(0.00000108629058104 + ((-0.24999998509883881) * (((4.0 * X1) * ((-1.0) * X1)) + ((-4.0) * X1))))", x)
    assert np.allclose(v, x * x + x, rtol=1e-5)
    assert np.allclose(S.evaluate_model("(exp(X1) - log(X1))", x), np.exp(x) - np.log(x))
    assert np.allclose(S.evaluate_model("(sin(X1) * cos(X1)) + sqrt(X1) + (X1 ^ 2.0)", x), np.sin(x) * np.cos(x) + np.sqrt(x) + x ** 2)


def test_structurally_exact_model_is_exact_after_refit():
    """A float32 scale and a tiny intercept around the exact structure: exact after the double-precision refit."""
    g = G.get("R-ARITH")
    tgt = G.evaluate(g, G.parse_expr("+(*(x, x), x)", g), G.grid_sample())
    scale = float(max(1.0, np.abs(tgt).max()))
    s = "(0.00000108629058104 + ((-0.24999998509883881) * (((4.0 * X1) * ((-1.0) * X1)) + ((-4.0) * X1))))"
    assert S.is_exact(s, tgt, scale)
    assert not S.is_exact("((-0.25) * ((4.0 * X1) * ((-1.0) * X1)))", tgt, scale)      # x^2 only: wrong structure


def test_hash_identical_to_the_engine():
    v = np.array([1.5 + 0j, 2.25, -3.0 + 1e-12j])
    assert S.scaled_hash(v, 3.0) == G.scaled_func_hash(v, 3.0)


@pytest.mark.skipif(not (ROOT / ".venv-ext/bin/python").exists() or os.environ.get("CI"), reason="needs .venv-ext")
def test_operon_run_end_to_end(tmp_path):
    g = G.get("R-ARITH4")                     # a calibration set (R-ARITH is held out and locked)
    tree = G.parse_expr("+(*(x, x), x)", g)
    v = G.evaluate(g, tree, G.grid_sample())
    t = dict(target_id="t0", expression="+(*(x, x), x)", K_construction=5, K_proven=5,
             target_scale=float(max(1.0, np.abs(v).max())), grid_values=[[float(z.real), float(z.imag)] for z in v])
    p = tmp_path / "t.json"
    p.write_text(json.dumps([t]))
    import hashlib
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(ROOT / ".venv-ext/operon-libs"), PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([str(ROOT / ".venv-ext/bin/python"), "-m", "reach.tasks.s2_operon", "--set", "R-ARITH4", "--targets",
                        str(p), "--target_id", "t0", "--targets_sha256", sha, "--config", "symbolic", "--budget", "1", "--seed", "1"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-500:]
    row = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT "))[7:])
    assert row["exact"] and row["evaluations"] <= 20_000 + 1000


def test_float_noise_in_the_imaginary_part_does_not_make_a_target_complex():
    """Review 2, finding 4: (log log log(x+2))^2 carries 3e-16 imaginary noise on the grid; the GP hash treats it as
    real, so an exactly right Operon model must score exact."""
    from reach.tasks.s2_operon_score import is_exact
    g = G.get("R-ELEM4")
    v = G.evaluate(g, G.parse_expr("exp(*(+(1, 1), log(log(log(log(+(x, +(1, 1))))))))", g), G.grid_sample())
    assert 0 < np.abs(v.imag).max() < 1e-12
    model = "(log(log(log((X1 + 2.0)))) * log(log(log((X1 + 2.0)))))"
    assert is_exact(model, v, float(max(1.0, np.abs(v).max())), G.grid_sample().real)


def test_a_model_that_cannot_be_evaluated_is_not_exact():
    """Guardrail row 74: a constant subexpression dividing by zero raised in plain Python and the run became an error
    row (s2_cal_operon, 2026-10-26); such a model is not the target, so it scores not exact."""
    from reach.tasks.s2_operon_score import evaluate_model, is_exact
    x = G.grid_sample().real
    assert not np.isfinite(evaluate_model("(X1 + (1.0 / 0.0))", x)).any()
    g = G.get("R-ARITH4")
    v = G.evaluate(g, G.parse_expr("+(x, 1)", g), G.grid_sample())
    assert is_exact("(X1 + (1.0 / 0.0))", v, 4.0, x) is False

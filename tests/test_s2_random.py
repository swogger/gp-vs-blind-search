"""T9 random sampling (written before the code). REQ-MEA-13."""
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from reach import grammar as G

ROOT = Path(__file__).resolve().parents[1]


def _targets(tmp_path, g, exprs):
    ts = []
    for i, e in enumerate(exprs):
        tree = G.parse_expr(e, g)
        t = dict(target_id=f"t{i}", expression=e)
        if g.kind == "real":
            v = G.evaluate(g, tree, G.grid_sample())
            t.update(var_target=float(np.var(v.real)), target_scale=float(max(1.0, np.abs(v).max())))
        ts.append(t)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(ts))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _slow_count(g, targets, n, seed):
    """Independent recount: regenerate the same trees and compare each with each target directly."""
    from reach.tasks.s2_random import draw_trees
    want = [G.parse_expr(t["expression"], g) for t in targets]
    tv = [G.evaluate(g, w, G.grid_sample()) if g.kind == "real" else G.evaluate(g, w) for w in want]
    hits = {t["target_id"]: 0 for t in targets}
    for tree in draw_trees(g, n, seed):
        if g.kind == "real":
            v = G.evaluate(g, tree, G.grid_sample())
            for t, w in zip(targets, tv):
                if np.isfinite(v).all() and np.allclose(v, w, rtol=1e-12, atol=1e-12):
                    hits[t["target_id"]] += 1
        else:
            v = G.evaluate(g, tree)
            for t, w in zip(targets, tv):
                hits[t["target_id"]] += v == w
    return hits


def _run(*args):
    p = subprocess.run([sys.executable, "-m", "reach.tasks.s2_random", *args], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))[7:])


def test_boolean_hits_match_independent_count(tmp_path):
    g = G.get("BOOL-NAND-n3")
    exprs = ["nand(x1, x2)", "nand(x1, x1)", "nand(nand(x1, x2), x3)"]
    p, sha = _targets(tmp_path, g, exprs)
    r = _run("--set", g.name, "--targets", str(p), "--targets_sha256", sha, "--n", "3000", "--seed", "5")
    assert r["n"] == 3000 and r["hits"] == _slow_count(g, json.loads(p.read_text()), 3000, 5)
    assert sum(r["hits"].values()) > 0


def test_real_hits_match_independent_count(tmp_path):
    g = G.get("R-ARITH4")
    p, sha = _targets(tmp_path, g, ["*(x, x)", "+(x, 1)", "/(x, +(x, 1))"])
    r = _run("--set", g.name, "--targets", str(p), "--targets_sha256", sha, "--n", "2000", "--seed", "3")
    assert r["hits"] == _slow_count(g, json.loads(p.read_text()), 2000, 3)
    assert 0 <= r["undefined_share"] <= 1 and 0 <= r["constant_share"] <= 1


def test_constant_multiple_is_not_a_hit(tmp_path):
    """Guardrail row 42: paper 1's random_sample hashed each tree with its own scale, so 2·x² counted as x²."""
    g = G.get("R-ARITH4")
    p, sha = _targets(tmp_path, g, ["*(x, x)"])
    from reach.tasks.s2_random import draw_trees
    twice = sum(1 for tree in draw_trees(g, 2000, 3)
                if G.expr_str(tree) in ("*(+(1, 1), *(x, x))", "*(*(x, 1), +(x, x))"))
    assert twice > 0                                                     # the trap is present in this draw
    r = _run("--set", g.name, "--targets", str(p), "--targets_sha256", sha, "--n", "2000", "--seed", "3")
    assert r["hits"] == _slow_count(g, json.loads(p.read_text()), 2000, 3)

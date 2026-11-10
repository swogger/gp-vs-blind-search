"""Stage 5 key sets (D125; docs/design/stage5_restarts.md): REQ-ENG-28. The counter's `record_keys` keeps every distinct
key a run evaluated that lies in the set's lookup, with the evaluation of its first sight, the run's own target left
out, and writes them to a side file; it changes nothing else (as REQ-ENG-23). Written before the code."""
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reach import grammar as G
from reach import revisit as V
from reach.engines import treegp as E
from reach.enumerate import Enumerator

ROOT = Path(__file__).resolve().parents[1]
TIMING = {"cpu_seconds", "wall_seconds"}
BOOL_TGT = "nand(nand(x1, nand(x2, x3)), nand(x3, nand(x1, x2)))"
REAL_TGT = "/(*(x, x), -(x, 1))"


@pytest.fixture(scope="module")
def enums(tmp_path_factory):
    out = {}
    for name, kmax in (("BOOL-NAND-n3", 9), ("R-ARITH4", 7)):
        g = G.get(name)
        d = tmp_path_factory.mktemp(name)
        Enumerator(g, d, log=lambda *_: None).run(kmax)
        out[name] = (g, d)
    return out


def _real_kw(g, t):
    grid, val = G.sample_grid(g), G.sample_val(g, 20260722)
    return dict(val_xs=val, val_target=G.evaluate(g, t, val), var_norm=0.023, grid_xs=grid,
                grid_target=G.evaluate(g, t, grid), grid_scale=4.49)


def _cases(enums):
    gb, db = enums["BOOL-NAND-n3"]
    gr, dr = enums["R-ARITH4"]
    tb, tr = G.parse_expr(BOOL_TGT, gb), G.parse_expr(REAL_TGT, gr)
    out = []
    for seed in (1, 2):
        out += [(gb, db, tb, "A", seed, dict(generations=25)), (gr, dr, tr, "A", seed, dict(generations=4,
                                                                                            **_real_kw(gr, tr)))]
    out.append((gb, db, tb, "A", 3, dict(generations=25, selection="none")))
    out.append((gb, db, G.parse_expr("nand(x1, nand(x2, x3))", gb), "A", 1, dict(generations=25)))   # hits
    return out


class Logged(V.Counter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trees = []

    def generation(self, population):
        self.trees.append([p.copy() for p in population])
        super().generation(population)


def _run(monkeypatch, case, record_keys, plant=False):
    g, d, t, regime, seed, kw = case
    made = []

    class Tracked(random.Random):
        def __init__(self, *a):
            super().__init__(*a)
            made.append(self)
    c = Logged(g, t, V.Lookup(g, d), pop_size=100, prefixes=(300, 600), record_keys=record_keys)
    random.seed(20261108)
    with monkeypatch.context() as m:
        m.setattr(random, "Random", Tracked)
        if plant:                                       # the recording consumes one number of the run's stream
            orig = c.generation
            m.setattr(c, "generation", lambda pop: (made[2].random(), orig(pop))[1])
        r = E.run_gp(g, t, regime, seed, stop_on="exact", timeout_seconds=600, counter=c, **kw)
    return r, [x.getstate() for x in made], random.getstate(), c


def _differences(a, b):
    out = [k for k in sorted((set(a[0]) | set(b[0])) - TIMING) if a[0].get(k) != b[0].get(k)]
    ra, rb = ({k: v for k, v in x[3].record().items() if k != "cpu_seconds"} for x in (a, b))
    if json.dumps(ra, sort_keys=True) != json.dumps(rb, sort_keys=True):
        out.append("rv")
    if a[1] != b[1]:
        out.append("generator states")
    if a[2] != b[2]:
        out.append("global random state")
    return out


def test_record_keys_changes_nothing(monkeypatch, enums):
    for case in _cases(enums):
        a, b = _run(monkeypatch, case, False), _run(monkeypatch, case, True)
        assert _differences(a, b) == [], (case[0].name, case[4])
        assert len(b[3].keys()[0]) > 0
    a = _run(monkeypatch, _cases(enums)[0], False)                       # plant: the check fires
    b = _run(monkeypatch, _cases(enums)[0], True, plant=True)
    assert "generator states" in _differences(a, b)


def _slow(g, d, target, trees, keep_target=False):
    """Independent log: the enumerator's own key function and linear level search (`min_size`) over every evaluated
    tree, in evaluation order; {key: evaluation of first sight} for keys with a level, the target left out."""
    e = Enumerator(g, d, log=lambda *_: None)
    e.load()
    xs = G.sample_grid(g) if g.kind == "real" else None

    def vec(t):
        return G.evaluate(g, t, xs) if g.kind == "real" else G.evaluate(g, t)

    def key(v):
        return int(e._key(v[None, :] if g.kind == "real" else np.array([v], dtype=np.uint64))[0])
    tkey, first, n = key(vec(target)), {}, 0
    for pop in trees:
        for t in pop:
            n += 1
            v = vec(t)
            k = key(v)
            if k in first or (k == tkey and not keep_target):
                continue
            if e.min_size(v) is not None:
                first[k] = n
    return first


def test_keys_equal_slow_log(monkeypatch, enums):
    hits = 0
    for case in _cases(enums):
        g, d, t = case[:3]
        r, _, _, c = _run(monkeypatch, case, True)
        idx, first = c.keys()
        got = {int(k): int(n) for k, n in zip(c.lookup.keys[idx], first)}
        assert got == _slow(g, d, t, c.trees), (g.name, case[4])
        assert list(first) == sorted(first)                                # in order of first sight
        if c.target_first is not None and c.lookup.levels([c.target])[0] > 0:
            hits += 1                                                      # plant: keeping the target fires
            assert got != _slow(g, d, t, c.trees, keep_target=True)
    assert hits >= 1


def test_side_file_is_deterministic(monkeypatch, enums, tmp_path):
    _, _, _, c = _run(monkeypatch, _cases(enums)[0], True)
    a = c.write_keys(tmp_path / "a" / "k.npz")
    b = c.write_keys(tmp_path / "b" / "k.npz")
    assert a["sha256"] == b["sha256"] == hashlib.sha256((tmp_path / "a/k.npz").read_bytes()).hexdigest()
    with np.load(tmp_path / "a/k.npz") as z:
        idx, first = c.keys()
        assert (z["idx"] == idx).all() and (z["first"] == first).all() and z["idx"].dtype == np.uint32
    assert a["n"] == len(idx) and not list(tmp_path.glob("*/*.tmp"))


def test_s2_run_keys_out_adds_only_its_field(tmp_path, enums):
    g, d = enums["BOOL-NAND-n3"]
    p = tmp_path / "targets.json"
    p.write_text(json.dumps([dict(target_id="t0", expression=BOOL_TGT, K_construction=9, K_proven=9,
                                  units=dict(nodes=9.0))]))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    args = ["--set", "BOOL-NAND-n3", "--targets", str(p), "--target_id", "t0", "--targets_sha256", sha,
            "--algorithm", "A", "--config", "base", "--seed", "7", "--generations", "60", "--pop_size", "100",
            "--penalty", "scaled", "--stop_on", "exact", "--seed_leaves", "off", "--timeout", "60",
            "--count_distinct", "--enum_dir", str(d)]

    def run(extra=(), ok=True):
        pr = subprocess.run([sys.executable, "-m", "reach.tasks.s2_run", *args, *extra], cwd=ROOT,
                            capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        assert (pr.returncode == 0) == ok, pr.stderr[-500:]
        return json.loads(next(l for l in pr.stdout.splitlines() if l.startswith("RESULT "))[7:]) if ok else pr
    out = tmp_path / "keys" / "t0_7.npz"
    a, b = run(), run(["--keys_out", str(out)])
    assert "keys" not in a and b["keys"]["file"] == str(out) and out.exists()
    assert b["keys"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest() and b["keys"]["n"] > 0
    strip = lambda r: {k: (dict(v, cpu_seconds=None) if k == "rv" else v)            # noqa: E731
                       for k, v in r.items() if k not in TIMING | {"keys"}}
    assert strip(a) == strip(b)
    pr = subprocess.run([sys.executable, "-m", "reach.tasks.s2_run", *args[:-3], "--keys_out", str(out)], cwd=ROOT,
                        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    assert pr.returncode != 0 and "refused" in pr.stdout + pr.stderr                # needs --count_distinct

"""Re-run counter for H6d (T25, D098; docs/design/h6_explanation.md): REQ-ENG-23 (the counter sees exactly the budgeted
evaluations, counts as the enumeration does and changes nothing else), REQ-ENG-24 (keys and levels). Written before the
code."""
import json
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
    """Small enumerations of one Boolean and one real grammar (the real ones of the study are untracked)."""
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
        out += [(gb, db, tb, "A", seed, dict(generations=25)),
                (gb, db, tb, "B", seed, dict(generations=25)),
                (gr, dr, tr, "A", seed, dict(generations=4, **_real_kw(gr, tr)))]
    return out


def _counter(g, d, t, prefixes=(300, 600, 1200), cls=V.Counter):
    return cls(g, t, V.Lookup(g, d), pop_size=100, prefixes=prefixes)


def _traced(monkeypatch, case, counter=None, plant=False):
    """One run; returns (result, final state of every random generator the run created, global random state)."""
    g, d, t, regime, seed, kw = case
    made = []

    class Tracked(random.Random):
        def __init__(self, *a):
            super().__init__(*a)
            made.append(self)

    random.seed(20261028)
    with monkeypatch.context() as m:
        m.setattr(random, "Random", Tracked)
        if plant:                                     # the counter consumes one random number of the run's stream
            orig = counter.generation
            m.setattr(counter, "generation", lambda pop: (made[-1].random(), orig(pop))[1])
        r = E.run_gp(g, t, regime, seed, stop_on="exact", timeout_seconds=600, counter=counter, **kw)
    return r, [x.getstate() for x in made], random.getstate()


def _differences(a, b):
    (ra, sa, ga), (rb, sb, gb) = a, b
    out = [k for k in sorted((set(ra) | set(rb)) - TIMING)
           if json.dumps(ra.get(k), default=str) != json.dumps(rb.get(k), default=str)]
    if sa != sb:
        out.append("generator states")
    if ga != gb:
        out.append("global random state")
    return out


def test_counter_changes_no_random_draw_and_no_other_field(monkeypatch, enums):
    for case in _cases(enums):
        g, d, t = case[:3]
        a = _traced(monkeypatch, case)
        b = _traced(monkeypatch, case, counter=_counter(g, d, t))
        assert len(a[1]) == 3                          # master, population and evolution generators all tracked
        assert _differences(a, b) == [], (g.name, case[3], case[4])


def test_counter_check_fires_when_the_counter_draws_a_random_number(monkeypatch, enums):
    """Validation of the check above on a planted violation."""
    case = _cases(enums)[0]
    g, d, t = case[:3]
    a = _traced(monkeypatch, case)
    b = _traced(monkeypatch, case, counter=_counter(g, d, t), plant=True)
    assert "generator states" in _differences(a, b)


def test_counter_time_does_not_count_against_the_soft_cpu_limit(monkeypatch, enums):
    """A fake CPU clock that only the counter advances: the search itself takes no time, so with the counter's time
    excluded the run never reaches its soft limit; counted, it would stop at the first check."""
    g, d = enums["BOOL-NAND-n3"]
    t = G.parse_expr(BOOL_TGT, g)
    clock = [0.0]
    monkeypatch.setattr(E.time, "process_time", lambda: clock[0])

    class Slow(V.Counter):
        calls = 0

        def generation(self, population):
            super().generation(population)
            clock[0] += 10.0                           # the counter "spends" 10 CPU seconds per generation
            self.calls += 1
    c = _counter(g, d, t, cls=Slow)
    r = E.run_gp(g, t, "B", 3, stop_on="exact", timeout_seconds=5.0, generations=8, counter=c)
    assert r["terminal_state"] != "timeout" and c.calls >= 2
    assert c.record()["cpu_seconds"] == pytest.approx(10.0 * c.calls)  # run_gp times the counter from outside


class Logged(V.Counter):
    """Keeps a copy of every tree it is shown, for the slow independent count."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trees = []

    def generation(self, population):
        self.trees.append([p.copy() for p in population])
        super().generation(population)


def test_counter_sees_exactly_the_budgeted_evaluations(monkeypatch, enums):
    seen_states = set()
    for case in _cases(enums):
        g, d, t = case[:3]
        c = _counter(g, d, t, cls=Logged)
        r = _traced(monkeypatch, case, counter=c)[0]
        rec = c.record()
        n = 100 * (r["gens_run"] + (r["terminal_state"] == "converged"))   # the evaluated generations (review m1:
        assert rec["stop"]["n"] == n == sum(len(p) for p in c.trees)     # run_gp's `evals` adds one generation for
        assert all(len(p) == 100 for p in c.trees)                        # runs that met the tolerance unexactly)
        for k, snap in rec["prefixes"].items():         # a prefix is recorded when the run reaches it, and only then
            assert snap["n"] == int(k) <= n
        assert set(rec["prefixes"]) == {str(p) for p in (300, 600, 1200) if p <= n}
        if g.kind == "bool":                            # a Boolean run first sees its target's key in its hit generation
            if r["terminal_state"] == "converged":
                assert r["gens_run"] * 100 < rec["target_first"] <= n
            else:
                assert rec["target_first"] is None
        seen_states.add(r["terminal_state"])
    assert seen_states == {"converged", "budget_exhausted"}


def _slow(g, d, target, trees, upto):
    """Independent count over the first `upto` evaluations: keys from the enumerator's own key function, distinct keys
    by a Python set, levels by the enumerator's linear search (`min_size`), undefined or constant by its own rule."""
    e = Enumerator(g, d, log=lambda *_: None)
    e.load()
    flat = [t for pop in trees for t in pop][:upto]
    xs = G.sample_grid(g) if g.kind == "real" else None

    def vec(t):
        return G.evaluate(g, t, xs) if g.kind == "real" else G.evaluate(g, t)

    def key(v):
        return int(e._key(v[None, :] if g.kind == "real" else np.array([v], dtype=np.uint64))[0])

    tkey = key(vec(target))
    keys, undef, first = [], 0, {}
    for t in flat:
        v = vec(t)
        k = key(v)
        keys.append(k)
        if g.kind == "real":
            with np.errstate(all="ignore"):
                undef += int(not (np.isfinite(v).all() and np.abs(v - v.mean()).max() > 1e-6))
        if k not in first:
            first[k] = v
    lv, lv_ok = [0] * (max(e.keys) + 1), [0] * (max(e.keys) + 1)
    for k, v in first.items():
        if k == tkey:
            continue
        m = e.min_size(v)
        lv[m if m is not None else 0] += 1
        with np.errstate(all="ignore"):                 # the enumerator's viability rule, on its own
            ok = g.kind != "real" or bool(e._viable(v[None, :])[0])
        lv_ok[m if m is not None else 0] += ok
    tfirst = next((i + 1 for i, k in enumerate(keys) if k == tkey), None)
    return dict(n=len(flat), d=len(set(keys)), undef=undef, lv=lv, lv_ok=lv_ok, target_first=tfirst)


def test_counts_equal_a_slow_independent_count(monkeypatch, enums):
    for case in _cases(enums):
        g, d, t = case[:3]
        c = _counter(g, d, t, cls=Logged)
        _traced(monkeypatch, case, counter=c)
        rec = c.record()
        for snap in list(rec["prefixes"].values()) + [rec["stop"]]:
            slow = _slow(g, d, t, c.trees, snap["n"])
            tf = rec["target_first"] if rec["target_first"] is not None and rec["target_first"] <= snap["n"] else None
            assert dict({k: snap[k] for k in ("n", "d", "undef", "lv", "lv_ok")}, target_first=tf) == slow, \
                (g.name, case[3], case[4], snap["n"])


def test_tree_keys_equal_the_enumeration_keys(enums):
    for g, d in enums.values():
        e = Enumerator(g, d, log=lambda *_: None)
        e.load()
        key = V.key_function(g)
        for k in sorted(e.keys):
            for idx in range(0, len(e.keys[k]), max(1, len(e.keys[k]) // 25)):
                assert key(G.parse_expr(e.expr(k, idx), g))[0] == int(e.keys[k][idx]), (g.name, k, idx)


def test_lookup_returns_each_level_and_zero_above(enums):
    for g, d in enums.values():
        e = Enumerator(g, d, log=lambda *_: None)
        e.load()
        lk = V.Lookup(g, d)
        assert lk.top == max(e.keys)
        for k in sorted(e.keys):
            assert (lk.levels(e.keys[k].astype(np.uint64)) == k).all()
        planted = np.setdiff1d(np.array([1, 2, 3, 2 ** 40 + 7], dtype=np.uint64), e.known)
        assert len(planted) and (lk.levels(planted) == 0).all()
        cum = {st["K"]: st["U_cum"] for st in e.stats}                # a bound below the ceiling, at a level after
        k0 = next(k for k in sorted(cum) if k >= 3 and cum.get(k + 1, 0) > cum[k])   # which new functions appear
        lo = V.Lookup(g, d, bound_bits=float(np.log2(cum[k0])))
        assert lo.top == k0 and len(e.keys[k0 + 1]) and (lo.levels(e.keys[k0 + 1].astype(np.uint64)) == 0).all()
        assert (lo.levels(e.keys[k0].astype(np.uint64)) == k0).all()


def _enum_dir(name):
    """The set's enumeration; in a second worktree its untracked level files live in the main one."""
    from reach.analysis.s2.phase1 import enum_dir
    d = enum_dir(name)
    if not (d / "level_001.npz").exists():
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
        d = (ROOT / common).resolve().parent / d.relative_to(ROOT)
    return d


@pytest.mark.study_machine
def test_calibration_targets_are_found_at_their_level():
    """Every calibration target below the bound is found at its proven level in its set's real enumeration (four sets,
    two per domain; the validation step checks all 16)."""
    for name in ("R-ARITH4", "F-EML", "BOOL-AON-n5", "BOOL-IMP0-n5"):
        g = G.get(name)
        lk = V.Lookup(g, _enum_dir(name))
        key = V.key_function(g)
        ts = json.loads((ROOT / f"data/targets_s2/{name}/calibration.json").read_text())
        below = [t for t in ts if t.get("K_proven") and t["K_proven"] <= lk.top]
        assert below
        got = lk.levels(np.array([key(G.parse_expr(t["expression"], g))[0] for t in below], dtype=np.uint64))
        assert list(got) == [t["K_proven"] for t in below], name


def test_s2_run_count_distinct_adds_only_its_field(tmp_path, enums):
    import hashlib
    g, d = enums["BOOL-NAND-n3"]
    p = tmp_path / "targets.json"
    p.write_text(json.dumps([dict(target_id="t0", expression=BOOL_TGT, K_construction=9, K_proven=9,
                                  units=dict(nodes=9.0))]))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    args = ["--set", "BOOL-NAND-n3", "--targets", str(p), "--target_id", "t0", "--targets_sha256", sha,
            "--algorithm", "A", "--config", "base", "--seed", "7", "--generations", "60", "--pop_size", "100",
            "--penalty", "scaled", "--stop_on", "exact", "--seed_leaves", "off", "--timeout", "60"]

    def run(extra=()):
        import os
        pr = subprocess.run([sys.executable, "-m", "reach.tasks.s2_run", *args, *extra], cwd=ROOT,
                            capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))  # row 88
        assert pr.returncode == 0, pr.stderr[-500:]
        return json.loads(next(l for l in pr.stdout.splitlines() if l.startswith("RESULT "))[7:])
    a, b = run(), run(["--count_distinct", "--enum_dir", str(d)])
    assert "rv" not in a and b["rv"]["stop"]["n"] == b["gens_reached"] * 100 + (100 if b["exact"] else 0)
    assert {k: v for k, v in a.items() if k not in TIMING} == {k: v for k, v in b.items() if k not in TIMING | {"rv"}}
    with pytest.raises(AssertionError):                                       # the option needs its enumeration
        run(["--count_distinct"])

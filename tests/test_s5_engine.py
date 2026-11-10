"""Stage 5 engine options (D125; docs/design/stage5_arms.md): REQ-ENG-25 (real_fitness), REQ-ENG-26 (bool_rows),
REQ-ENG-27 (selection "none"), REQ-ENG-29 (wider batch interval) and REQ-ENG-30 (configurations, s2_run). Every
option defaults to the current engine byte for byte; new draws come from the run's own generators, never from pop_rng
or evo_rng. Written before the code; every check is shown to fire on a planted violation."""
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reach import grammar as G
from reach.engines import treegp as E

ROOT = Path(__file__).resolve().parents[1]
TIMING = {"cpu_seconds", "wall_seconds"}
BOOL3 = "nand(nand(x1, nand(x2, x3)), nand(x3, nand(x1, x2)))"
BOOL3_B = "nand(nand(x1, x2), nand(x3, nand(x3, x1)))"
BOOL5 = "nand(nand(x1, nand(x2, x5)), nand(x4, nand(x3, x2)))"
BOOL5_B = "nand(nand(x2, nand(x3, x1)), nand(x5, nand(nand(x4, x2), x1)))"
REAL = "/(*(x, x), -(x, 1))"                           # R-ARITH4
REAL_B = "+(*(x, *(x, x)), /(x, +(x, 1)))"


def _real(expr=REAL, name="R-ARITH4"):
    g = G.get(name)
    t = G.parse_expr(expr, g)
    grid, val = G.sample_grid(g), G.sample_val(g, 20260722)
    v = G.evaluate(g, t, grid)
    var = float(np.mean(np.abs(v - v.mean()) ** 2))
    return g, t, dict(val_xs=val, val_target=G.evaluate(g, t, val), var_norm=var, grid_xs=grid, grid_target=v,
                      grid_scale=float(max(1.0, np.abs(v).max())), undefined_penalty="scaled")


def _bool(expr=BOOL3, name="BOOL-NAND-n3"):
    g = G.get(name)
    return g, G.parse_expr(expr, g), {}


def _traced(monkeypatch, g, t, seed, plant=None, **kw):
    """One run; returns (result, final states of the generators the run created, global random state, generators)."""
    made = []

    class Tracked(random.Random):
        def __init__(self, *a):
            super().__init__(*a)
            made.append(self)

    random.seed(20261108)
    with monkeypatch.context() as m:
        m.setattr(random, "Random", Tracked)
        if plant:
            plant(m, made)
        r = E.run_gp(g, t, "A", seed, stop_on="exact", timeout_seconds=600, **kw)
    return r, [x.getstate() for x in made], random.getstate()


def _fields(r):
    return {k: v for k, v in r.items() if k not in TIMING}


def _same(a, b, n_states=None):
    """Differences between two traced runs: result fields, then the states of the first n_states generators."""
    out = [k for k in sorted(set(a[0]) | set(b[0]) - TIMING) if k not in TIMING
           and json.dumps(a[0].get(k), default=str) != json.dumps(b[0].get(k), default=str)]
    sa, sb = (a[1], b[1]) if n_states is None else (a[1][:n_states], b[1][:n_states])
    if sa != sb:
        out.append("generator states")
    if a[2] != b[2]:
        out.append("global random state")
    return out


def _cases():
    out = []
    for seed in (1, 2):
        out.append(_bool() + (seed, dict(generations=30)))
        out.append(_real() + (seed, dict(generations=5)))
    return out


# ---------------------------------------------------------------- REQ-ENG-25 real_fitness
def test_real_fitness_default_is_bit_identical(monkeypatch):
    """Every new option at its default equals the engine without it: fields, all generator states, global state."""
    for g, t, kw, seed, run in _cases():
        a = _traced(monkeypatch, g, t, seed, **kw, **run)
        b = _traced(monkeypatch, g, t, seed, real_fitness="batch", bool_rows=None, batch_domain=None,
                    batch_var_norm=None, selection="tournament", **kw, **run)
        assert len(a[1]) == 3 and _same(a, b) == [], (g.name, seed)
    g, t, kw = _real()                                        # the wider interval set to (2, 3) is the base too
    a = _traced(monkeypatch, g, t, 3, generations=5, **kw)
    b = _traced(monkeypatch, g, t, 3, generations=5, batch_domain=(G.DOMAIN_LO, G.DOMAIN_HI),
                batch_var_norm=kw["var_norm"], **kw)
    assert _same(a, b) == []


def test_frozen_first_generation_equals_batch(monkeypatch):
    g, t, kw = _real()
    for seed in (1, 2, 3):
        a = _traced(monkeypatch, g, t, seed, generations=1, **kw)
        b = _traced(monkeypatch, g, t, seed, generations=1, real_fitness="frozen", **kw)
        assert _same(a, b) == [], seed


def _points_used(monkeypatch, g, t, seed, **kw):
    """The distinct point sets on which trees were evaluated during one run (as sorted tuples of the real parts),
    and how often the target was evaluated."""
    seen, target_calls = set(), [0]
    real_eval = G.evaluate

    def spy(grammar, tree, xs=None):
        if xs is not None:
            seen.add(tuple(np.round(np.sort(np.real(np.asarray(xs))), 15)))
            target_calls[0] += tree is t
        return real_eval(grammar, tree, xs)
    with monkeypatch.context() as m:
        m.setattr(G, "evaluate", spy)
        r = E.run_gp(g, t, "A", seed, stop_on="exact", timeout_seconds=600, **kw)
    return r, seen, target_calls[0]


def _key(xs):
    return tuple(np.round(np.sort(np.real(np.asarray(xs))), 15))


def _grid_violations(seen, kw):
    """Point sets other than the identity grid and the validation sample (the check of 'grid' fitness)."""
    return seen - {_key(kw["grid_xs"]), _key(kw["val_xs"])}


def test_grid_error_uses_the_identity_grid(monkeypatch):
    g, t, kw = _real()
    r, seen, calls = _points_used(monkeypatch, g, t, 4, generations=6, real_fitness="grid", **kw)
    assert r["gens_run"] >= 1 and _grid_violations(seen, kw) == set() and calls == 0
    _, seen, calls = _points_used(monkeypatch, g, t, 4, generations=6, **kw)            # plant: the fresh batch
    assert len(_grid_violations(seen, kw)) >= 2 and calls >= 2
    r, seen, calls = _points_used(monkeypatch, g, t, 4, generations=6, real_fitness="frozen", **kw)
    assert len(_grid_violations(seen, kw)) == 1 and calls == 1                          # generation 0's batch only


def test_real_only_options_are_refused_on_boolean_sets():
    g, t, _ = _bool()
    for kw in (dict(real_fitness="grid"), dict(real_fitness="frozen"), dict(batch_domain=(1.5, 3.5)),
               dict(batch_var_norm=2.0)):
        with pytest.raises(ValueError):
            E.run_gp(g, t, "A", 1, generations=2, **kw)
    gr, tr, kw = _real()
    with pytest.raises(ValueError):
        E.run_gp(gr, tr, "A", 1, generations=2, bool_rows=24, **kw)
    with pytest.raises(ValueError):
        E.run_gp(gr, tr, "A", 1, generations=2, real_fitness="noisy", **kw)


# ---------------------------------------------------------------- REQ-ENG-26 bool_rows
def test_all_rows_equals_full_table_run(monkeypatch):
    """bool_rows = all rows: the same run as the default (the mask's generator never touches evo_rng); a planted mask
    drawn from evo_rng changes the run."""
    for (g, t, _), rows in ((_bool(), 8), (_bool(BOOL5, "BOOL-NAND-n5"), 32)):
        for seed in (1, 2):
            a = _traced(monkeypatch, g, t, seed, generations=40)
            b = _traced(monkeypatch, g, t, seed, generations=40, bool_rows=rows)
            assert len(b[1]) == 4 and _same(a, b, n_states=3) == [], (g.name, seed)
    g, t, _ = _bool(BOOL5, "BOOL-NAND-n5")
    orig = E._row_mask

    def plant(m, made):                                   # the mask drawn from the evolution generator (made[2])
        m.setattr(E, "_row_mask", lambda rng, rows, k: orig(made[2], rows, k))
    a = _traced(monkeypatch, g, t, 1, generations=40)
    b = _traced(monkeypatch, g, t, 1, generations=40, bool_rows=32, plant=plant)
    assert _same(a, b, n_states=3) != []


def test_mask_rows_distinct_and_fresh(monkeypatch):
    g, t, _ = _bool(BOOL5, "BOOL-NAND-n5")
    masks = []
    orig = E._row_mask

    def spy(rng, rows, k):
        m = orig(rng, rows, k)
        masks.append(m)
        return m
    monkeypatch.setattr(E, "_row_mask", spy)
    E.run_gp(g, t, "A", 5, generations=30, stop_on="exact", bool_rows=24)
    first = list(masks)
    assert len(first) >= 2 and all(bin(m).count("1") == 24 and m < (1 << 32) for m in first)
    assert len(set(first)) == len(first)                  # a fresh draw each generation
    masks.clear()
    E.run_gp(g, t, "A", 5, generations=30, stop_on="exact", bool_rows=24)
    assert masks == first                                 # seeded by the run's seed
    masks.clear()
    E.run_gp(g, t, "A", 6, generations=30, stop_on="exact", bool_rows=24)
    assert masks[:3] != first[:3]


def test_elite_uses_full_table():
    """With one row per generation many trees tie at zero error; the elite, the recorded error and the stop rule
    still use the full truth table, so every stop is an exact match and final_nmse is the full-table error."""
    g, t, _ = _bool(BOOL5, "BOOL-NAND-n5")
    tt = G.evaluate(g, t)
    for k in (1, 24):
        for seed in (1, 2, 3):
            r = E.run_gp(g, t, "A", seed, generations=40, stop_on="exact", bool_rows=k, full_expr=True)
            best = G.parse_expr(r["best_expr_full"], g)
            assert r["final_nmse"] == bin(G.evaluate(g, best) ^ tt).count("1")
            assert (r["terminal_state"] == "converged") == r["exact_hash_match"] == (r["final_nmse"] == 0)


# ---------------------------------------------------------------- REQ-ENG-27 selection "none"
class _Log:
    """A counter stand-in that keeps every evaluated generation's trees (as expressions)."""
    def __init__(self):
        self.gens, self.cpu = [], 0.0

    def generation(self, population):
        self.gens.append([G.expr_str(p) for p in population])


def _blind_differences(g, ta, tb, seed, kw_a=None, kw_b=None, generations=12):
    """Generations (before either run stops) whose evaluated trees differ between two targets."""
    la, lb = _Log(), _Log()
    ra = E.run_gp(g, ta, "A", seed, generations=generations, stop_on="exact", selection="none", counter=la,
                  **(kw_a or {}))
    rb = E.run_gp(g, tb, "A", seed, generations=generations, stop_on="exact", selection="none", counter=lb,
                  **(kw_b or {}))
    n = min(len(la.gens), len(lb.gens))
    assert n >= 3, "a run stopped too early to test"
    return [i for i in range(n) if la.gens[i] != lb.gens[i]], ra, rb


def test_selection_free_is_target_blind():
    g = G.get("BOOL-NAND-n5")
    for seed in (1, 2):
        diff, ra, _ = _blind_differences(g, G.parse_expr(BOOL5, g), G.parse_expr(BOOL5_B, g), seed)
        assert diff == []
        assert ra["evals"] == 100 * (ra["gens_run"] + (ra["terminal_state"] == "converged"))
    g, ta, kw_a = _real(REAL)
    _, tb, kw_b = _real(REAL_B)
    diff, _, _ = _blind_differences(g, ta, tb, 3, kw_a, kw_b, generations=6)
    assert diff == []


def test_kept_elite_plant_fails(monkeypatch):
    """Planted: the elite slot keeps the best so far (selection through the elite), so the runs see their targets."""
    monkeypatch.setattr(E, "_selfree_elite", lambda population, best_tree, rng: best_tree.copy())
    g = G.get("BOOL-NAND-n5")
    diff, _, _ = _blind_differences(g, G.parse_expr(BOOL5, g), G.parse_expr(BOOL5_B, g), 1)
    assert diff != []


def test_tournament_of_one_is_not_target_blind():
    """Guardrail row 172: the cost probe's stand-in (tournament size 1, elite kept) is not target-blind, because a
    uniform index into the error-sorted population picks a tree that depends on the target."""
    g = G.get("BOOL-NAND-n5")
    la, lb = _Log(), _Log()
    for t, log in ((BOOL5, la), (BOOL5_B, lb)):
        E.run_gp(g, G.parse_expr(t, g), "A", 1, generations=6, stop_on="exact", tournament_size=1, counter=log)
    assert any(a != b for a, b in zip(la.gens, lb.gens))


def test_selection_free_draws_neither_population_nor_evolution_stream_for_selection(monkeypatch):
    """The selection-free run creates one generator more than the base run (its own), and the batch points of the
    real domain are those of the base run in generation 0 (drawn from evo_rng before any selection)."""
    g, t, kw = _real()
    a = _traced(monkeypatch, g, t, 1, generations=1, **kw)
    b = _traced(monkeypatch, g, t, 1, generations=1, selection="none", **kw)
    assert len(b[1]) == 4 and b[1][:2] == a[1][:2]        # master and population generators untouched


# ---------------------------------------------------------------- REQ-ENG-29 the wider batch interval
def test_wider_interval_batch_points_and_exactness(monkeypatch):
    g, t, kw = _real()
    r, seen, _ = _points_used(monkeypatch, g, t, 2, generations=6, batch_domain=(1.5, 3.5),
                              batch_var_norm=3 * kw["var_norm"], **kw)
    batches = _grid_violations(seen, kw)
    assert len(batches) >= 2
    pts = np.array([x for b in batches for x in b])
    assert pts.min() > 1.5 and pts.max() < 3.5 and (pts < 2).any() and (pts > 3).any()
    exact = E.run_gp(g, t, "A", 1, generations=1, stop_on="exact", batch_domain=(1.5, 3.5),
                     batch_var_norm=kw["var_norm"], **kw)
    assert isinstance(exact["exact_hash_match"], bool)    # exactness still decided on the (2, 3) grid


def _first_batch(monkeypatch, g, t, seed, **kw):
    """Generation 0's batch points, in order (the first evaluation of the target)."""
    got = []
    real_eval = G.evaluate

    def spy(grammar, tree, xs=None):
        if tree is t and not got:
            got.append(np.real(np.asarray(xs)).copy())
        return real_eval(grammar, tree, xs)
    with monkeypatch.context() as m:
        m.setattr(G, "evaluate", spy)
        E.run_gp(g, t, "A", seed, stop_on="exact", timeout_seconds=600, generations=1, **kw)
    return got[0]


def test_wider_interval_keeps_the_stream_position(monkeypatch):
    """The batch takes the same uniform draws from evo_rng: generation 0's wide points are the base points mapped
    from (2, 3) onto (1.5, 3.5) (the same stream position; only the interval differs)."""
    g, t, kw = _real()
    for seed in (5, 6):
        base = _first_batch(monkeypatch, g, t, seed, **kw)
        wide = _first_batch(monkeypatch, g, t, seed, batch_domain=(1.5, 3.5), batch_var_norm=kw["var_norm"], **kw)
        assert len(base) == len(wide) == E.BATCH_SIZE
        assert np.allclose(wide, 1.5 + 2.0 * (base - 2.0), rtol=0, atol=1e-12)


# ---------------------------------------------------------------- REQ-ENG-30 configurations and s2_run
def test_stage5_configurations():
    from reach.tasks.s2_configs import CONFIGS, WIDE_INTERVAL
    assert CONFIGS["FG"]["kwargs"] == dict(real_fitness="grid") and CONFIGS["FG"]["domain"] == "real"
    assert CONFIGS["FB"]["kwargs"] == dict(real_fitness="frozen") and CONFIGS["FB"]["domain"] == "real"
    assert CONFIGS["BS24"]["kwargs"] == dict(bool_rows=24) and CONFIGS["BS24"]["domain"] == "bool"
    assert CONFIGS["SF"]["kwargs"] == dict(selection="none") and "domain" not in CONFIGS["SF"]
    assert CONFIGS["WIDE"]["kwargs"] == dict(batch_domain=WIDE_INTERVAL) and CONFIGS["WIDE"]["domain"] == "real"
    assert WIDE_INTERVAL == (1.5, 3.5)
    for name in ("FG", "FB", "BS24", "SF", "WIDE"):
        assert CONFIGS[name]["pop_size"] == 100 and CONFIGS[name]["generations_per_E0"] == 200
    for name in ("base", "R1", "R2", "R3", "R4", "R5", "R6"):
        assert "domain" not in CONFIGS[name]


def _task_args(tmp_path, s, expr, config, extra=()):
    t = dict(target_id="t0", expression=expr, K_construction=7, K_proven=7, units=dict(nodes=7.0))
    if G.get(s).kind == "real":
        g = G.get(s)
        v = G.evaluate(g, G.parse_expr(expr, g), G.sample_grid(g))
        t.update(var_target=float(np.mean(np.abs(v - v.mean()) ** 2)), target_scale=float(max(1, np.abs(v).max())))
    p = tmp_path / f"{s}.json"
    p.write_text(json.dumps([t]))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return ["--set", s, "--targets", str(p), "--target_id", "t0", "--targets_sha256", sha, "--algorithm", "A",
            "--config", config, "--seed", "7", "--generations", "3", "--pop_size", "100", "--penalty", "scaled",
            "--stop_on", "exact", "--seed_leaves", "off", "--timeout", "60", *extra]


def _s2_run(args):
    import os
    return subprocess.run([sys.executable, "-m", "reach.tasks.s2_run", *args], cwd=ROOT, capture_output=True,
                          text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))


def test_s2_run_refuses_a_configuration_of_the_other_domain(tmp_path):
    for s, expr, cfg in (("BOOL-NAND-n3", "nand(x1, x2)", "FG"), ("BOOL-NAND-n3", "nand(x1, x2)", "FB"),
                         ("BOOL-NAND-n3", "nand(x1, x2)", "WIDE"), ("R-ARITH4", "*(x, x)", "BS24")):
        p = _s2_run(_task_args(tmp_path, s, expr, cfg))
        assert p.returncode != 0 and "refused" in (p.stdout + p.stderr), (s, cfg)
    for s, expr, cfg in (("R-ARITH4", "*(x, +(x, 1))", "FG"), ("R-ARITH4", "*(x, +(x, 1))", "WIDE"),
                         ("BOOL-NAND-n5", "nand(x1, nand(x2, x3))", "BS24"), ("BOOL-NAND-n3", "nand(x1, x2)", "SF"),
                         ("R-ARITH4", "*(x, +(x, 1))", "SF")):
        p = _s2_run(_task_args(tmp_path, s, expr, cfg))
        assert p.returncode == 0, (s, cfg, p.stderr[-400:])


def test_s2_run_wide_config_normalises_by_the_wide_variance(tmp_path, monkeypatch):
    """W15's batch errors are normalised by the target's variance on the wide interval's 100-point grid (finite
    points), so the scale-aware penalty keeps its meaning there (D034)."""
    from reach.tasks import s2_configs as C
    g = G.get("R-ARITH4")
    t = G.parse_expr("/(*(x, x), -(x, 1))", g)
    xs = C.interval_grid(*C.WIDE_INTERVAL)
    v = G.evaluate(g, t, xs)
    assert len(xs) == G.N_POINTS and xs.real.min() > 1.5 and xs.real.max() < 3.5
    assert C.wide_var_norm(g, t) == pytest.approx(float(np.mean(np.abs(v - v.mean()) ** 2)))
    planted = G.parse_expr("log(-(x, 2))", G.get("R-ELEM6"))   # undefined on part of the wide interval
    w = C.wide_var_norm(G.get("R-ELEM6"), planted)
    assert np.isfinite(w) and w > 0


@pytest.mark.parametrize("opts", [dict(real_fitness="grid"), dict(real_fitness="frozen"),
                                  dict(batch_domain=(1.5, 3.5)), dict(selection="none")])
def test_prefix_property_holds_for_the_real_options(opts):
    """REQ-ENG-03 for the real-valued Stage 5 options: a run of k generations is the start of a longer one."""
    g, t, kw = _real("+(*(x, x), /(x, +(x, 1)))")
    if "batch_domain" in opts:
        opts = dict(opts, batch_var_norm=4 * kw["var_norm"])
    for seed in range(1, 9):
        full = E.run_gp(g, t, "A", seed, generations=40, stop_on="exact", **opts, **kw)
        if full["converged"] and full["gens_run"] >= 1:
            k = full["gens_run"]
            assert not E.run_gp(g, t, "A", seed, generations=k, stop_on="exact", **opts, **kw)["converged"]
            just = E.run_gp(g, t, "A", seed, generations=k + 1, stop_on="exact", **opts, **kw)
            assert just["converged"] and just["best_expr"] == full["best_expr"]
            return
    pytest.fail(f"no converging seed for {opts}")


def _uniform_draws(monkeypatch, g, t, seed, **kw):
    """Number of uniform draws from each generator the run created (galib never draws uniform numbers, so the
    evolution generator's are the batch points)."""
    counts = []

    class Counted(random.Random):
        def __init__(self, *a):
            super().__init__(*a)
            counts.append(0)
            self._i = len(counts) - 1

        def uniform(self, a, b):
            counts[self._i] += 1
            return super().uniform(a, b)
    with monkeypatch.context() as m:
        m.setattr(random, "Random", Counted)
        r = E.run_gp(g, t, "A", seed, stop_on="exact", timeout_seconds=600, **kw)
    return r, counts


@pytest.mark.parametrize("opts", [dict(), dict(real_fitness="grid"), dict(real_fitness="frozen"),
                                  dict(batch_domain=(1.5, 3.5)), dict(selection="none")])
def test_every_mode_draws_the_batch(monkeypatch, opts):
    """The batch is drawn from evo_rng in every generation of every mode (draw and discard in grid and frozen)."""
    g, t, kw = _real("+(*(x, x), /(x, +(x, 1)))")
    for seed in (1, 6):
        r, counts = _uniform_draws(monkeypatch, g, t, seed, generations=8, **opts, **kw)
        evaluated = r["gens_run"] + (r["terminal_state"] == "converged")
        assert counts[2] == E.BATCH_SIZE * evaluated and sum(counts) == counts[2], (opts, seed)


# ---------------------------------------------------------------- review F1 (guardrail row 173)
PLANTED_WIDE = ("F-EML", "F-EML_K19_03")       # finite on 81 of 100 wide-grid points, values up to 1.7e214


def _calibration_target(g_name, tid):
    t = next(x for x in json.loads((ROOT / f"data/targets_s2/{g_name}/calibration.json").read_text())
             if x["target_id"] == tid)
    g = G.get(g_name)
    return g, G.parse_expr(t["expression"], g), t


def test_batch_var_norm_must_be_finite_and_positive():
    """A normaliser of inf made every batch error NaN (mse / inf with an infinite penalty): no selection, and only the
    elite copy was ever checked, so the run could not succeed. run_gp refuses it."""
    g, t, kw = _real()
    for bad in (float("inf"), float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError):
            E.run_gp(g, t, "A", 1, generations=2, batch_domain=(1.5, 3.5), batch_var_norm=bad, **kw)
    E.run_gp(g, t, "A", 1, generations=1, batch_domain=(1.5, 3.5), batch_var_norm=2.0, **kw)   # a finite one runs


def test_wide_normaliser_of_the_planted_target_is_not_finite():
    from reach.tasks.s2_configs import wide_ok, wide_var_norm
    g, tree, _ = _calibration_target(*PLANTED_WIDE)
    assert wide_var_norm(g, tree) == float("inf") and not wide_ok(g, tree)
    g2, t2, _ = _real()
    assert wide_ok(g2, t2)


def test_s2_run_refuses_a_wide_run_whose_normaliser_is_not_finite():
    g, _, t = _calibration_target(*PLANTED_WIDE)
    rel = f"data/targets_s2/{g.name}/calibration.json"
    sha = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    args = ["--set", g.name, "--targets", rel, "--target_id", t["target_id"], "--targets_sha256", sha,
            "--algorithm", "A", "--config", "WIDE", "--seed", "7", "--generations", "2", "--pop_size", "100",
            "--penalty", "scaled", "--stop_on", "exact", "--seed_leaves", "off", "--timeout", "60"]
    p = _s2_run(args)
    assert p.returncode != 0 and "refused" in p.stdout + p.stderr and "normaliser" in p.stdout + p.stderr
    p = _s2_run(args[:args.index("WIDE")] + ["base"] + args[args.index("WIDE") + 1:])
    assert p.returncode == 0, p.stderr[-300:]                              # the base run on it is unchanged

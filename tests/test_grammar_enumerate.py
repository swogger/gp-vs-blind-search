"""Grammar layer, M1 exactness and the M4 enumerator: positive and negative controls."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from reach import grammar as G
from reach.engines.treegp import run_gp
from reach.enumerate import Enumerator

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "legacy/results/synthetic_dense_k/targets.json"


def test_syntactic_counts_match_eml_table_c1():
    assert G.syntactic_counts(G.get("EML"), 7)[1::2] == [2, 4, 16, 80]


def test_parse_print_roundtrip():
    for name, s in [("EML", "eml(eml(1, eml(x, x)), 1)"), ("ELEM6", "+(exp(x), log(/(x, 1)))"),
                    ("BOOL-AON-n3", "or(and(x1, not(x2)), x3)"), ("BOOL-AX1-n3", "xor(x1, 1)")]:
        assert G.expr_str(G.parse_expr(s, G.get(name))) == s


def test_ieee_division():
    g = G.get("ELEM6")
    v = G.evaluate(g, G.parse_expr("/(x, -(x, x))", g), G.grid_sample())
    assert not np.isfinite(v).any()


def _real_kwargs(g, tree, tgt):
    grid, val = G.grid_sample(), G.validation_sample(20260721 + 1)
    return dict(val_xs=val, val_target=G.evaluate(g, tree, val), var_norm=tgt["var_target"],
                grid_xs=grid, grid_target=G.evaluate(g, tree, grid), grid_scale=tgt["target_scale"])


@pytest.mark.study_machine
def test_legacy_mode_reproduces_p1_row():
    g = G.get("EML")
    tgt = next(t for t in json.loads(TARGETS.read_text()) if t["target_id"] == "K07_01")
    tree = G.parse_expr(tgt["expression"], g)
    r = run_gp(g, tree, "A", 1753007184, legacy=True, **_real_kwargs(g, tree, tgt))
    assert (r["converged"], r["gens_to_converge"], r["final_tree_size"], r["final_nmse"]) == (True, 22, 7, 0.0)


def test_no_size_penalty_exact_solution_converges():
    g = G.get("EML")                                   # target x: the seeded leaf is exact at gen 0
    tree = G.leaf("x")
    tgt = dict(var_target=float(np.var(G.grid_sample().real)), target_scale=2.5)
    r = run_gp(g, tree, "A", 1, legacy=True, generations=5, **_real_kwargs(g, tree, tgt))
    assert r["converged"] and r["gens_to_converge"] == 0


def test_boolean_gp_runs_and_converges_on_easy_target():
    g = G.get("BOOL-NAND-n3")
    r = run_gp(g, G.parse_expr("nand(x1, x2)", g), "A", 3, generations=50)
    assert r["converged"] and r["exact_hash_match"]


def _brute_min_sizes(g, kmax):
    """Minimal size of every function by explicit tree enumeration (independent code path)."""
    a0 = [G.leaf(v) for v in list(g.variables) + list(g.constants)]
    trees = {1: a0}
    best = {}
    for k in range(1, kmax + 1):
        if k > 1:
            trees[k] = [G.Node(o.name, t, None, True) for o in g.ops if o.arity == 1 for t in trees[k - 1]]
            trees[k] += [G.Node(o.name, l, r, True) for o in g.ops if o.arity == 2
                         for i in range(1, k - 1) for l in trees[i] for r in trees[k - 1 - i]]
        for t in trees[k]:
            best.setdefault(G.evaluate(g, t), k)
    return best


@pytest.mark.parametrize("name", ["BOOL-NAND-n3", "BOOL-AON-n3", "BOOL-AX1-n3"])
def test_boolean_enumeration_matches_brute_force(name, tmp_path):
    g = G.get(name)
    kmax = 7
    e = Enumerator(g, tmp_path, log=lambda *_: None)
    e.run(kmax)
    brute = _brute_min_sizes(g, kmax)
    ours = {int(key): k for k in e.keys for key in e.keys[k]}
    assert ours == brute


def test_candidate_counts_are_products_of_level_sizes(tmp_path):
    g = G.get("EML")
    e = Enumerator(g, tmp_path, log=lambda *_: None)
    e.run(9)
    syn = G.syntactic_counts(g, 9)
    for st in e.stats[1:]:                     # every candidate is one syntactic tree of that size
        k = st["K"]
        assert st["candidates"] == sum(len(e.keys[i]) * len(e.keys[k - 1 - i]) for i in range(1, k - 1))
    assert e.stats[0]["U"] == syn[1] == 2


def test_eml_proven_targets_appear_first_at_their_size(tmp_path):
    g = G.get("EML")
    e = Enumerator(g, tmp_path, log=lambda *_: None)
    e.run(11)
    grid = G.grid_sample()
    targets = [t for t in json.loads(TARGETS.read_text()) if t["K_proven"] not in ("", None)]
    assert targets
    for t in targets:
        assert e.min_size(G.evaluate(g, G.parse_expr(t["expression"], g), grid)) == t["K_proven"], t["target_id"]
    # negative control: a function absent up to K = 11 is not found
    assert e.min_size(grid * 1e3 + 0.123) is None


def test_resume_rebuilds_identical_levels(tmp_path):
    g = G.get("ELEM6")
    e1 = Enumerator(g, tmp_path, log=lambda *_: None)
    e1.run(5)
    e2 = Enumerator(g, tmp_path, log=lambda *_: None)
    e2.run(6)
    e3 = Enumerator(g, tmp_path / "fresh", log=lambda *_: None)
    e3.run(6)
    for k in range(1, 7):
        assert np.array_equal(e2.keys[k], e3.keys[k])
        assert np.array_equal(e2.vals[k], e3.vals[k], equal_nan=True)


def test_stop_on_exact_continues_past_loose_convergence():
    """exp(exp(x)) in F-EML: loose solutions like exp(e^x) - ln x meet the tolerance; with stop_on='exact'
    a run ends only on an exact match (or the budget) and never reports a loose solution as converged-exact."""
    g = G.get("F-EML")
    tree = G.parse_expr("eml(eml(x, 1), 1)", g)
    grid, val = G.grid_sample(), G.validation_sample(20260721 + 1)
    v = G.evaluate(g, tree, grid)
    kw = dict(val_xs=val, val_target=G.evaluate(g, tree, val), var_norm=float(np.mean(np.abs(v - v.mean()) ** 2)),
              grid_xs=grid, grid_target=v, grid_scale=float(np.abs(v).max()), generations=60)
    for seed in range(4):
        r = run_gp(g, tree, "A", seed, stop_on="exact", **kw)
        assert (r["terminal_state"] == "converged") == r["exact_hash_match"]

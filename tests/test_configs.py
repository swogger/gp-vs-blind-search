"""T19 GP configurations R1-R6 (written before the code). REQ-ENG-04, REQ-ENG-09, REQ-ENG-03."""
import json
import random

import numpy as np
import pytest

from reach import grammar as G
from reach.engines import treegp as E
from reach.engines.treegp import run_gp

TGT = "nand(nand(x1, nand(x2, x3)), nand(x3, nand(x1, x2)))"


def _bool(expr=TGT, name="BOOL-NAND-n3"):
    g = G.get(name)
    return g, G.parse_expr(expr, g)


def test_defaults_are_bit_identical_to_the_base_configuration():
    g, t = _bool()
    for seed in (1, 2, 3):
        a = run_gp(g, t, "A", seed, generations=60, stop_on="exact")
        b = run_gp(g, t, "A", seed, generations=60, stop_on="exact", tournament_size=5, selection="tournament",
                   p_crossover=0.75, p_mutation=0.20, init="ramped", max_depth=E.MAX_DEPTH,
                   real_fitness="batch", bool_rows=None, batch_domain=None, batch_var_norm=None)   # Stage 5 defaults
        for k in ("best_expr", "final_nmse", "gens_run", "terminal_state", "final_tree_size"):
            assert a[k] == b[k]


@pytest.mark.parametrize("opts", [dict(tournament_size=2), dict(selection="lexicase"), dict(init="uniform"),
                                  dict(max_depth=17), dict(p_crossover=0.9, p_mutation=0.1),
                                  dict(bool_rows=6), dict(selection="none")])                  # Stage 5 (D125)
def test_prefix_property_holds_for_every_option(opts):
    """REQ-ENG-03: a shorter run is the start of a longer one, for every configuration."""
    g, t = _bool()
    for seed in range(1, 8):
        full = run_gp(g, t, "A", seed, generations=150, stop_on="exact", **opts)
        if full["converged"]:
            k = full["gens_run"]
            assert not run_gp(g, t, "A", seed, generations=k, stop_on="exact", **opts)["converged"]
            just = run_gp(g, t, "A", seed, generations=k + 1, stop_on="exact", **opts)
            assert just["converged"] and just["best_expr"] == full["best_expr"]
            return
    pytest.fail(f"no converging seed for {opts}")


def test_lexicase_selects_by_cases():
    """On a hand-made table the elite on the first case wins when ties are broken by later cases."""
    errs = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0]], float)
    rng = random.Random(0)
    picks = {E.lexicase_select(errs, np.zeros(3), rng) for _ in range(50)}
    assert picks == {3}                                               # individual 3 is best on every case
    errs2 = np.array([[0, 5], [5, 0]], float)
    picks2 = {E.lexicase_select(errs2, np.zeros(2), random.Random(s)) for s in range(30)}
    assert picks2 == {0, 1}                                           # case order decides between specialists


def test_epsilon_lexicase_keeps_near_best_real_cases():
    errs = np.array([[0.0, 1.0], [0.01, 0.0], [5.0, 0.0]])
    eps = np.array([0.1, 0.1])
    picks = {E.lexicase_select(errs, eps, random.Random(s)) for s in range(40)}
    assert 1 in picks                                                 # within eps of the best on case 1


def test_uniform_initialisation_draws_sizes_uniformly():
    g = G.get("BOOL-AON-n3")
    pop = E.uniform_population(g, 3000, E.MAX_DEPTH, random.Random(1))
    sizes = [G.count_nodes(t) for t in pop]
    assert max(sizes) <= 31 and all(G.tree_depth(t) <= E.MAX_DEPTH for t in pop)
    counts = np.bincount(sizes)
    present = counts[counts > 0]
    assert present.max() / present.min() < 1.6                        # roughly uniform over available sizes


def test_depth_17_allows_deeper_trees():
    g, t = _bool()
    r = run_gp(g, t, "A", 4, generations=40, stop_on="exact", max_depth=17)
    assert r["terminal_state"] in ("converged", "budget_exhausted")


def test_s2_run_knows_every_configuration():
    from reach.tasks import s2_configs as C
    assert set(C.CONFIGS) == {"base", "R1", "R2", "R3", "R4", "R5", "R6",
                              "FG", "FB", "BS24", "SF", "WIDE"}                                # Stage 5 (D125)
    assert C.CONFIGS["R1"]["pop_size"] == 25 and C.CONFIGS["R1"]["generations_per_E0"] == 800
    assert C.CONFIGS["R2"]["pop_size"] == 1000 and C.CONFIGS["R2"]["generations_per_E0"] == 20
    for name, c in C.CONFIGS.items():
        assert c["pop_size"] * c["generations_per_E0"] == 20_000, name


def test_depth_limit_17_does_not_change_initialisation(monkeypatch):
    """Review 2, finding 1 (guardrail row 64): R6 changes only the depth limit (one factor at a time). Its initial
    population must be grown to the base depth (MAX_DEPTH = 10), as in every other configuration."""
    from reach.engines import treegp as T
    seen = []
    real = T.generate_ramped_population
    monkeypatch.setattr(T, "generate_ramped_population", lambda n, d, *a, **k: seen.append(d) or real(n, d, *a, **k))
    g = G.get("BOOL-NAND-n3")
    tree = G.parse_expr("nand(x1, x2)", g)
    for depth in (T.MAX_DEPTH, 17):
        T.run_gp(g, tree, "A", 7, generations=2, pop_size=20, legacy=False, max_depth=depth, timeout_seconds=30)
    assert seen == [T.MAX_DEPTH, T.MAX_DEPTH]

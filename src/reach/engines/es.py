"""(1 + lambda) evolution strategy on trees: a second search family for the EML-family study (D023).

One parent; each generation lambda children by subtree mutation (galib, same operator and depth limit as
tree GP); the best child replaces the parent if its training error is lower, or equal with no more nodes
(neutral moves accepted, but a lexicographic size tie-break stops neutral drift into bloat; it never
trades error for size, so it is not a size penalty, docs/07 #1). Training error on a fresh batch each generation, best-so-far tracked on the
validation sample, stop on an exact match (or the budget). Budget E = evaluations = lambda x generations.
No crossover, no population: if node reach is invariant here too, the size law is not GP-specific.
"""
import math
import random as _random_module
import time

import numpy as np

from reach import grammar as G
from reach.engines.treegp import BATCH_SIZE, CONV_THRESHOLD, MAX_DEPTH, _PATCHED, _patch
from reach.galib.evolution import mutate
from reach.galib.generator import generate_random_tree

_INF = float("inf")


def run_es(g, target_tree, seed, *, val_xs, val_target, var_norm, grid_xs, grid_target, grid_scale,
           evaluations=20_000, lam=4, timeout_seconds=600.0, undefined_penalty="fixed"):
    t0, c0 = time.time(), time.process_time()
    saved = {n: getattr(_random_module, n) for n in _PATCHED}
    penalty = {"fixed": 1000.0, "scaled": 1000.0 * var_norm}[undefined_penalty]   # as run_gp (D034)

    def error(tree, xs, target):
        if G.count_nodes(tree) > G.MAX_NODES:
            return _INF
        return G.mse_against(G.evaluate(g, tree, xs), target, penalty) / var_norm

    def is_exact(tree):
        return (G.scaled_func_hash(G.evaluate(g, tree, grid_xs), grid_scale)
                == G.scaled_func_hash(grid_target, grid_scale))

    def batch(rng):
        draw = lambda: np.array([complex(rng.uniform(G.DOMAIN_LO, G.DOMAIN_HI)) for _ in range(BATCH_SIZE)])
        return draw() if len(g.variables) == 1 else {v: draw() for v in g.variables}

    consts, vars_ = list(g.constants), list(g.variables)
    mut_depth = max(2, round(math.sqrt(MAX_DEPTH)))
    generations = evaluations // lam
    try:
        rng = _random_module.Random(seed)
        _patch(rng)
        parent = generate_random_tree(rng.randint(2, 6), g.operators, consts, vars_, rng.choice(["full", "grow"]))
        best_tree, best_val, gens_to_converge = parent, error(parent, val_xs, val_target), None
        converged, state, gens_run = False, "budget_exhausted", generations
        for gen in range(generations):
            if time.process_time() - c0 > timeout_seconds:
                state, gens_run = "timeout", gen
                break
            bx = batch(rng)
            bt = G.evaluate(g, target_tree, bx)
            p_err = error(parent, bx, bt)
            kids = []
            for _ in range(lam):
                c = mutate(parent, g.operators, consts, vars_, mut_depth, absolute_max_depth=MAX_DEPTH)
                kids.append((error(c, bx, bt) if G.count_nodes(c) <= G.MAX_NODES else _INF, c))
            k_err, kid = min(kids, key=lambda x: (x[0], G.count_nodes(x[1])))
            if k_err < p_err or (k_err == p_err and G.count_nodes(kid) <= G.count_nodes(parent)):
                parent = kid
            v = error(parent, val_xs, val_target)
            if v < best_val:
                best_val, best_tree = v, parent
            if best_val <= CONV_THRESHOLD:
                if gens_to_converge is None:
                    gens_to_converge = gen
                if is_exact(best_tree):
                    converged, state, gens_run = True, "converged", gen
                    break
    finally:
        for n, f in saved.items():
            setattr(_random_module, n, f)
    size = G.count_nodes(best_tree)
    return {"converged": converged or best_val <= CONV_THRESHOLD, "exact_hash_match": bool(is_exact(best_tree)),
            "gens_to_converge": gens_to_converge, "final_nmse": best_val, "final_tree_size": size,
            "terminal_state": state, "gens_run": gens_run, "evals": lam * (gens_run + 1 if converged else gens_run),
            "wall_seconds": time.time() - t0, "cpu_seconds": time.process_time() - c0,
            "best_expr": G.expr_str(best_tree) if size <= 200 else ""}

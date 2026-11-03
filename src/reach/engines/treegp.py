"""Tree GP for any grammar: legacy/research/synthetic/eml_synth_core.run_gp generalised (M1).

`legacy=True` reproduces EML-GP exactly: depth limit on initialisation and mutation only,
the seeded single-variable leaf, early stopping after 600 stagnant generations. The study
defaults (D010) are `legacy=False`: the same depth limit on crossover too, and no early
stopping. There is no size penalty in either mode (docs/07 #1). RNG consumption follows
EML-GP's order so the EML regression is exact; galib uses the global `random` module, so
run one GP run per process.
"""
import math
import random as _random_module
import time

import numpy as np

from reach import grammar as G
from reach.galib.evolution import crossover, mutate, point_mutate
from reach.galib.generator import generate_ramped_population, generate_random_tree

MAX_DEPTH = math.ceil(math.log2(G.MAX_NODES + 1)) - 1   # 10
BATCH_SIZE = 75
POP_SIZE = 100
GENERATIONS = 200
CONV_THRESHOLD = 1e-6
LEGACY_PLATEAU = 600
EXPR_MAX_NODES = 200        # best_expr is printed only for best trees up to this size; full_expr adds the whole tree
_INF = float("inf")
_PATCHED = ("choice", "random", "sample", "uniform")


def _patch(rng):
    for name in _PATCHED:
        setattr(_random_module, name, getattr(rng, name))


def lexicase_select(case_errors, eps, rng):
    """Lexicase selection (epsilon-lexicase when eps > 0): filter the population case by case in random order,
    keeping those within eps of the best on each case; ties at the end are broken at random. Returns an index."""
    cand = np.arange(case_errors.shape[0])
    order = list(range(case_errors.shape[1]))
    rng.shuffle(order)
    for j in order:
        col = case_errors[cand, j]
        cand = cand[col <= col.min() + eps[j]]
        if len(cand) == 1:
            return int(cand[0])
    return int(cand[rng.randrange(len(cand))])


def uniform_population(g, n, max_depth, rng, max_size=31):
    """Initial population of trees with sizes drawn uniformly from the available sizes 1..max_size (R5), each tree
    uniform among the trees of that size within the depth limit (the exact sampler)."""
    from reach.sample import make_sampler
    sample = make_sampler(g)
    sizes = [k for k in range(1, max_size + 1) if sample.count(k, max_depth) > 0]
    return [sample(rng.choice(sizes), max_depth, rng) for _ in range(n)]


def run_gp(g, target_tree, regime, seed, *, val_xs=None, val_target=None, var_norm=1.0,
           grid_xs=None, grid_target=None, grid_scale=1.0, generations=GENERATIONS,
           pop_size=POP_SIZE, timeout_seconds=600.0, legacy=False, seed_leaves=None,
           point_mutation=False, stop_on="tolerance", undefined_penalty="fixed", crossover_mode="subtree",
           tournament_size=5, selection="tournament", p_crossover=0.75, p_mutation=0.20, init="ramped",
           max_depth=MAX_DEPTH, full_expr=False, counter=None):
    """One run. regime 'A' = crossover 0.75 + subtree mutation 0.20; 'B' = mutation only.
    timeout_seconds is CPU time. For Boolean grammars the data arguments are ignored.
    stop_on: 'tolerance' stops at validation nMSE <= 1e-6 (EML-GP); 'exact' keeps searching until the best
    tree is an exact match on the definition grid (a loose convergence no longer ends the run, D022).
    undefined_penalty: 'fixed' charges 1000 per non-finite residual before dividing by var_norm (EML-GP); 'scaled'
    charges 1000 * var_norm, i.e. 1000 after normalisation, so an undefined tree never beats a constant (D034).
    crossover_mode: 'subtree' (standard); 'headless' crosses each selected parent with a fresh random tree drawn
    like the initial population (headless-chicken control, Jones 1995; D034).
    Study 2 configurations (T19; R1-R6): tournament_size, selection ("tournament" or "lexicase": per-case errors, with
    epsilon = the per-case median absolute deviation for real data and 0 for Boolean), p_crossover, p_mutation,
    init ("ramped" or "uniform") and max_depth. The defaults reproduce the base configuration bit for bit.
    full_expr (reruns, D091): also return `best_expr_full`, the whole best tree, printed after the search with the
    random generators already restored, so the search and every other field are unchanged (REQ-ENG-22).
    counter (H6d re-runs, D098): a `reach.revisit.Counter` shown every evaluated generation; it draws no random number,
    and its CPU time, measured here around each call, is left out of the soft limit, so the search stops where it would
    without it (REQ-ENG-23)."""
    seed_leaves = legacy if seed_leaves is None else seed_leaves
    plateau = LEGACY_PLATEAU if legacy else None
    xdepth = None if legacy else max_depth
    mutation_only = regime == "B"
    is_bool = g.kind == "bool"
    t0, c0 = time.time(), time.process_time()
    saved = {n: getattr(_random_module, n) for n in _PATCHED}

    if is_bool:
        tt = G.evaluate(g, target_tree)

        def error(tree, *_):
            if G.count_nodes(tree) > G.MAX_NODES:
                return _INF
            return float(bin(G.evaluate(g, tree) ^ tt).count("1"))

        rows = 1 << g.n_inputs

        def case_errors(tree, *_):
            if G.count_nodes(tree) > G.MAX_NODES:
                return np.ones(rows)
            d = G.evaluate(g, tree) ^ tt
            return np.array([(d >> r) & 1 for r in range(rows)], float)
        threshold = 0.0
    else:
        def error(tree, xs, target):
            if G.count_nodes(tree) > G.MAX_NODES:
                return _INF
            return G.mse_against(G.evaluate(g, tree, xs), target, penalty) / var_norm

        def case_errors(tree, xs, target):
            if G.count_nodes(tree) > G.MAX_NODES:
                return np.full(len(target), penalty / var_norm)
            with np.errstate(all="ignore"):
                d = np.abs(G.evaluate(g, tree, xs) - target)
                bad = ~np.isfinite(d)
                return np.where(bad, penalty, np.where(bad, 0.0, d) ** 2) / var_norm
        penalty = {"fixed": 1000.0, "scaled": 1000.0 * var_norm}[undefined_penalty]
        threshold = CONV_THRESHOLD

    def is_exact(tree):
        if is_bool:
            return G.evaluate(g, tree) == tt
        return grid_xs is not None and (G.scaled_func_hash(G.evaluate(g, tree, grid_xs), grid_scale)
                                        == G.scaled_func_hash(grid_target, grid_scale))

    try:
        m = _random_module.Random(seed)
        m.randint(0, 2 ** 31 - 1)                     # drawn and unused, as in EML-GP
        pop_rng = _random_module.Random(m.randint(0, 2 ** 31 - 1))
        evo_rng = _random_module.Random(m.randint(0, 2 ** 31 - 1))
        _patch(pop_rng)
        init_depth = min(max_depth, MAX_DEPTH)          # R6 raises the limit only; trees start as in base (row 64)
        if init == "uniform":
            population = uniform_population(g, pop_size, init_depth, pop_rng)
        else:
            population = generate_ramped_population(pop_size, init_depth, g.operators, list(g.constants),
                                                    list(g.variables), seed_variable_leaves=seed_leaves)
        _patch(evo_rng)
        mut_depth = max(2, round(math.sqrt(MAX_DEPTH)))

        def random_mate():               # as generate_ramped_population: depth 2..MAX_DEPTH, full or grow
            return generate_random_tree(evo_rng.randint(2, init_depth), g.operators, list(g.constants),
                                        list(g.variables), evo_rng.choice(["full", "grow"]))
        best_tree, best_val, stagnant = None, _INF, 0
        converged, gens_to_converge, state, gens_run = False, None, "budget_exhausted", generations

        for gen in range(generations):
            if time.process_time() - c0 - (counter.cpu if counter is not None else 0.0) > timeout_seconds:
                state, gens_run = "timeout", gen
                break
            if is_bool:
                bx = b_target = None
            else:
                if len(g.variables) == 1:      # EML-GP's draw order (legacy exactness)
                    bx = np.array([complex(evo_rng.uniform(G.DOMAIN_LO, G.DOMAIN_HI))
                                   for _ in range(BATCH_SIZE)], dtype=np.complex128)
                else:
                    bx = {v: np.array([complex(evo_rng.uniform(G.DOMAIN_LO, G.DOMAIN_HI))
                                       for _ in range(BATCH_SIZE)], dtype=np.complex128) for v in g.variables}
                b_target = G.evaluate(g, target_tree, bx)
            scores = [(ind, error(ind, bx, b_target)) for ind in population]
            if counter is not None:
                t_c = time.process_time()
                counter.generation(population)
                counter.cpu += time.process_time() - t_c
            scores.sort(key=lambda item: item[1])
            if selection == "lexicase":
                cases = np.array([case_errors(ind, bx, b_target) for ind, _ in scores])
                eps = (np.zeros(cases.shape[1]) if is_bool else
                       np.median(np.abs(cases - np.median(cases, axis=0)), axis=0))
            gen_val = error(scores[0][0], val_xs, val_target)
            if gen_val < best_val:
                best_val, best_tree, stagnant = gen_val, scores[0][0], 0
            else:
                stagnant += 1
                if best_tree is None:   # every validation error so far NaN/inf: the old code crashed here (D027 A3)
                    best_tree = scores[0][0]
            if best_val <= threshold:
                if gens_to_converge is None:
                    gens_to_converge = gen
                if stop_on == "tolerance" or is_exact(best_tree):
                    converged, state, gens_run = True, "converged", gen
                    break
            if plateau is not None and stagnant >= plateau:
                state, gens_run = "early_stopped", gen
                break

            def pick():
                if selection == "lexicase":
                    return scores[lexicase_select(cases, eps, evo_rng)][0]
                tt_ = evo_rng.sample(scores, k=tournament_size)
                tt_.sort(key=lambda x: x[1])
                return tt_[0][0]

            new_pop = [best_tree.copy()]
            while len(new_pop) < pop_size:
                p1 = pick()
                if not mutation_only and evo_rng.random() < p_crossover and len(new_pop) < pop_size - 1:
                    p2 = pick()
                    if crossover_mode == "headless":
                        c1 = crossover(p1, random_mate(), max_depth=xdepth)[0]
                        c2 = crossover(p2, random_mate(), max_depth=xdepth)[0]
                    else:
                        c1, c2 = crossover(p1, p2, max_depth=xdepth)
                    if G.count_nodes(c1) > G.MAX_NODES:
                        c1 = p1.copy()
                    if G.count_nodes(c2) > G.MAX_NODES:
                        c2 = p2.copy()
                    new_pop.extend([c1, c2])
                else:
                    new_pop.append(p1.copy())

            consts, vars_ = list(g.constants), list(g.variables)
            for i in range(1, len(new_pop)):
                if mutation_only:
                    if point_mutation and evo_rng.random() < 0.05:
                        new_pop[i] = point_mutate(new_pop[i], g.operators, consts, vars_)
                    else:
                        cand = mutate(new_pop[i], g.operators, consts, vars_, mut_depth,
                                      absolute_max_depth=max_depth)
                        new_pop[i] = cand if G.count_nodes(cand) <= G.MAX_NODES else new_pop[i]
                else:
                    if evo_rng.random() < p_mutation:
                        cand = mutate(new_pop[i], g.operators, consts, vars_, mut_depth,
                                      absolute_max_depth=max_depth)
                        new_pop[i] = cand if G.count_nodes(cand) <= G.MAX_NODES else new_pop[i]
                    elif point_mutation and evo_rng.random() < 0.05:
                        new_pop[i] = point_mutate(new_pop[i], g.operators, consts, vars_)
            population = new_pop[:pop_size]
    finally:
        for n, f in saved.items():
            setattr(_random_module, n, f)

    if best_tree is None:
        best_tree = population[0]
        best_val = error(best_tree, val_xs, val_target)
    converged = converged or best_val <= threshold
    exact = is_exact(best_tree)
    size = G.count_nodes(best_tree)
    out = {"converged": converged, "exact_hash_match": bool(exact),
           "gens_to_converge": gens_to_converge, "final_nmse": best_val,
           "final_tree_size": size, "terminal_state": state, "gens_run": gens_run,
           "evals": pop_size * (gens_run + 1 if converged else gens_run),
           "wall_seconds": time.time() - t0, "cpu_seconds": time.process_time() - c0,
           "best_expr": G.expr_str(best_tree) if size <= EXPR_MAX_NODES else ""}
    if full_expr:
        out["best_expr_full"] = G.expr_str(best_tree)
    return out

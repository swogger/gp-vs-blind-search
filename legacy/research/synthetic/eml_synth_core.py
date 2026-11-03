"""Core library for the Dense-K synthetic-target experiment.

Design contract
---------------
The *search* is the main study's EML-GP verbatim: the population is built by
`ga-lib/generator.generate_ramped_population`, varied by
`ga-lib/evolution.{crossover,mutate,point_mutate}`, and the reproduction loop
below is a line-for-line transcription of `research/deduce_formula.py` at the
reference cell (pop 100, gen 200, node cap 2000).  Two deliberate,
documented deviations:

  1. Fitness is *normalised*: nMSE = MSE / Var(target).  The reference cell's
     absolute thresholds (parsimony 1e-5, convergence 1e-6) are meaningful only
     when the target is O(1).  Synthetic EML targets span many orders of
     magnitude, so without normalisation the same thresholds would mean wildly
     different things per target.  Normalisation is a constant rescaling, so it
     leaves selection order untouched; only the two thresholds change meaning,
     and they change it to the pre-registered one.

  2. Evaluation is vectorised over the sample with numpy instead of looping
     `cmath` per point.  `check_evaluator_agreement()` validates this against
     `ga-lib/node.Node.evaluate` on randomly generated trees; the agreement
     statistics are written into run_metadata.json.

Everything else — RNG stream derivation, ramped init, tournament k=5, 75%
crossover / 20% subtree mutation (regime A), 95/5 mutation split (regime B),
elitism, node cap, plateau rule — is unchanged.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys
import time

import numpy as np

_GA_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ga-lib"))
if _GA_LIB not in sys.path:
    sys.path.insert(0, _GA_LIB)

from node import Node                                    # noqa: E402
from generator import generate_ramped_population         # noqa: E402
from evolution import crossover, mutate, point_mutate    # noqa: E402

# --------------------------------------------------------------------------
# Reference-cell configuration (research/deduce_formula.py + run_matrix.py)
# --------------------------------------------------------------------------
MAX_NODES        = 2_000
MAX_DEPTH        = math.ceil(math.log2(MAX_NODES + 1)) - 1   # 10
BATCH_SIZE       = 75
POP_SIZE         = 100
GENERATIONS      = 200
TIMEOUT_SECONDS  = 600.0
# eml_plateau_mult(['+','-','*','/','exp','log']) == 12 for the Phase-1 EML
# runs; patience = max(50, 200//10) * 12 = 600 > 200 generations, i.e. plateau
# stopping never fires at the reference cell.  Kept verbatim.
PLATEAU_MULT     = 12
PLATEAU_GENS     = max(50, GENERATIONS // 10) * PLATEAU_MULT

DOMAIN_LO        = 2.0
DOMAIN_HI        = 3.0
N_POINTS         = 100
CONV_THRESHOLD   = 1e-6
HASH_PRECISION   = 9

OPERATORS        = ["eml"]
TERMINALS        = [1.0]
VARIABLES        = ["x"]

_INF = float("inf")


# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------
def leaf(v):
    return Node(v, is_operator=False)


def eml_node(a, b):
    return Node("eml", left=a, right=b, is_operator=True)


def count_nodes(tree):
    """Iterative twin of deduce_formula.count_nodes (same value, less overhead)."""
    if tree is None:
        return 0
    n = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        n += 1
        if node.is_operator:
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
    return n


def tree_depth(tree):
    """Same convention as ga-lib/evolution.get_depth (a bare leaf has depth 1)."""
    if tree is None:
        return 0
    return 1 + max(tree_depth(tree.left), tree_depth(tree.right))


def n_operators(tree):
    return (count_nodes(tree) - 1) // 2


def expr_str(tree):
    """Canonical `eml(a, b)` rendering with `x` / `1` leaves."""
    if not tree.is_operator:
        if isinstance(tree.value, str):
            return tree.value
        v = tree.value
        return str(int(v)) if float(v).is_integer() else repr(v)
    return f"eml({expr_str(tree.left)}, {expr_str(tree.right)})"


def parse_expr(s):
    """Inverse of expr_str, for the forced-inclusion target."""
    s = s.strip()
    if s.startswith("eml("):
        inner = s[4:-1]
        depth = 0
        for i, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                return eml_node(parse_expr(inner[:i]), parse_expr(inner[i + 1:]))
        raise ValueError(f"unbalanced: {s}")
    if s == "x":
        return leaf("x")
    return leaf(float(s))


# --------------------------------------------------------------------------
# Uniform sampling of depth-bounded EML trees
# --------------------------------------------------------------------------
_COUNT_CACHE = {}


def _tree_counts(max_m, max_h):
    """T[m][h] = number of binary tree *shapes* with m internal nodes and
    ga-lib depth <= h (a bare leaf has depth 1)."""
    key = (max_m, max_h)
    if key in _COUNT_CACHE:
        return _COUNT_CACHE[key]
    T = [[0] * (max_h + 1) for _ in range(max_m + 1)]
    for h in range(1, max_h + 1):
        T[0][h] = 1
    for m in range(1, max_m + 1):
        for h in range(2, max_h + 1):
            T[m][h] = sum(T[i][h - 1] * T[m - 1 - i][h - 1] for i in range(m))
    _COUNT_CACHE[key] = T
    return T


def sample_tree(m, max_h, rng):
    """Uniform over {EML trees with exactly m operators, depth <= max_h,
    leaves i.i.d. uniform on {x, 1}}."""
    T = _tree_counts(m, max_h)

    def build(mm, hh):
        if mm == 0:
            return leaf("x") if rng.random() < 0.5 else leaf(1.0)
        total = T[mm][hh]
        r = rng.randrange(total)
        acc = 0
        for i in range(mm):
            w = T[i][hh - 1] * T[mm - 1 - i][hh - 1]
            acc += w
            if r < acc:
                return eml_node(build(i, hh - 1), build(mm - 1 - i, hh - 1))
        raise AssertionError("weight underflow")

    return build(m, max_h)


# --------------------------------------------------------------------------
# Vectorised evaluation
# --------------------------------------------------------------------------
def vec_eval(tree, xs):
    """Evaluate an eml/{x,const} tree over a complex128 sample vector.

    Mirrors ga-lib/node.Node.evaluate: eml(a, b) = exp(a) - log(b) on the
    principal branch, with log(0) = -inf (numpy's own value, identical to the
    reference's explicit special case).  Overflow yields inf/nan rather than
    raising; the reference caught the equivalent OverflowError and returned
    complex(inf, inf).  Both are non-finite, and non-finiteness is absorbing
    under eml, so every downstream fitness is identical -- see
    check_evaluator_agreement().
    """
    with np.errstate(all="ignore"):
        return _vec_eval(tree, xs)


def _vec_eval(tree, xs):
    if not tree.is_operator:
        if isinstance(tree.value, str):
            return xs
        return np.full(xs.shape, complex(tree.value), dtype=np.complex128)
    a = _vec_eval(tree.left, xs)
    b = _vec_eval(tree.right, xs)
    return np.exp(a) - np.log(b)


def mse_against(pred, target):
    """deduce_formula._mse, vectorised: non-finite residuals cost 1000 each."""
    with np.errstate(all="ignore"):
        diff = np.abs(pred - target)
        bad = ~np.isfinite(diff)
        sq = np.where(bad, 0.0, diff) ** 2
        return float((np.where(bad, 1000.0, sq)).sum() / diff.size)


def target_variance(target):
    """Var(target) = mean |t - mean(t)|^2 (real, works for complex targets)."""
    with np.errstate(all="ignore"):
        return float(np.mean(np.abs(target - target.mean()) ** 2))


def func_hash(vec, precision=HASH_PRECISION):
    """Absolute round-to-`precision` hash; same convention as
    eml_core.minimality._hash_arr, so novelty and the minimality audit agree."""
    with np.errstate(all="ignore"):
        r = np.round(np.asarray(vec, dtype=np.complex128).real, precision)
        i = np.round(np.asarray(vec, dtype=np.complex128).imag, precision)
    return hashlib.sha1(r.tobytes() + i.tobytes()).hexdigest()


def scaled_func_hash(vec, scale, precision=HASH_PRECISION):
    """Scale-aware identity test used for `exact_hash_match`: the candidate's
    output must agree with the target to `precision` digits *relative to the
    target's magnitude*, which is what "same function" means for targets whose
    values are O(1e8)."""
    with np.errstate(all="ignore"):
        v = np.asarray(vec, dtype=np.complex128) / scale
        r = np.round(v.real, precision)
        i = np.round(v.imag, precision)
        r[~np.isfinite(r)] = np.nan
        i[~np.isfinite(i)] = np.nan
    return hashlib.sha1(r.tobytes() + i.tobytes()).hexdigest()


def grid_sample():
    """The N=100 evenly spaced definition sample on (2, 3), endpoints excluded."""
    step = (DOMAIN_HI - DOMAIN_LO) / (N_POINTS + 1)
    return np.array(
        [DOMAIN_LO + step * (i + 1) for i in range(N_POINTS)], dtype=np.complex128
    )


def validation_sample(seed):
    """Held-out N=100 sample: uniform draws on (2, 3) from a dedicated stream,
    fixed per target (identical across regimes and seeds, so final_nmse is
    comparable across every run of a target) and disjoint from the definition
    grid with probability 1."""
    rng = random.Random(seed)
    return np.array(
        [complex(rng.uniform(DOMAIN_LO, DOMAIN_HI)) for _ in range(N_POINTS)],
        dtype=np.complex128,
    )


# --------------------------------------------------------------------------
# The GP run
# --------------------------------------------------------------------------
def run_gp(
    target_tree,
    regime,
    seed,
    val_xs,
    val_target,
    var_norm,
    grid_xs=None,
    grid_target=None,
    grid_scale=1.0,
    generations=GENERATIONS,
    pop_size=POP_SIZE,
    timeout_seconds=TIMEOUT_SECONDS,
    size_penalty=True,      # legacy: add 1e-4*nodes below nMSE 1e-5
    point_mutation=False,   # legacy: no point mutation for the EML alphabet
    cpu_clock=False,        # legacy: timeout measured in wall-clock seconds
    plateau_gens=None,      # legacy: module constant PLATEAU_GENS (600)
):
    """One EML-GP run.  regime 'A' = crossover+mutation, 'B' = mutation only."""
    mutation_only = regime == "B"
    t0 = time.time()
    c0 = time.process_time()
    _plateau = PLATEAU_GENS if plateau_gens is None else plateau_gens

    # --- RNG streams: identical derivation to deduce_formula.main() ---------
    _m = random.Random(seed)
    _val_seed = _m.randint(0, 2 ** 31 - 1)      # drawn (unused) to keep the stream aligned
    pop_seed = _m.randint(0, 2 ** 31 - 1)
    evo_seed = _m.randint(0, 2 ** 31 - 1)
    pop_rng = random.Random(pop_seed)
    evo_rng = random.Random(evo_seed)

    import random as _random_module
    _random_module.choice = pop_rng.choice
    _random_module.random = pop_rng.random
    _random_module.sample = pop_rng.sample
    _random_module.uniform = pop_rng.uniform

    population = generate_ramped_population(
        pop_size, MAX_DEPTH, OPERATORS, TERMINALS, VARIABLES
    )

    _random_module.choice = evo_rng.choice
    _random_module.random = evo_rng.random
    _random_module.sample = evo_rng.sample
    _random_module.uniform = evo_rng.uniform

    def nmse(tree, xs, target):
        if count_nodes(tree) > MAX_NODES:
            return _INF
        return mse_against(vec_eval(tree, xs), target) / var_norm

    def training_fitness(tree, xs, target):
        f = nmse(tree, xs, target)
        if size_penalty and f < 1e-5:
            return f + count_nodes(tree) * 0.0001
        return f

    mut_depth = max(2, round(math.sqrt(MAX_DEPTH)))
    use_point_mutate = point_mutation   # legacy False; point_mutate also swaps leaves

    best_tree = None
    best_val = _INF
    gens_no_improve = 0
    converged = False
    gens_to_converge = None
    terminal_state = "budget_exhausted"
    gens_run = 0

    for gen in range(generations):
        gens_run = gen
        if ((time.process_time() - c0) if cpu_clock else (time.time() - t0)) > timeout_seconds:
            terminal_state = "timeout"
            break

        bx = np.array(
            [complex(evo_rng.uniform(DOMAIN_LO, DOMAIN_HI)) for _ in range(BATCH_SIZE)],
            dtype=np.complex128,
        )
        b_target = vec_eval(target_tree, bx)

        fitness_scores = [(ind, training_fitness(ind, bx, b_target)) for ind in population]
        fitness_scores.sort(key=lambda item: item[1])

        gen_best_tree = fitness_scores[0][0]
        gen_val_err = nmse(gen_best_tree, val_xs, val_target)

        if gen_val_err < best_val:
            best_val = gen_val_err
            best_tree = gen_best_tree
            gens_no_improve = 0
        else:
            gens_no_improve += 1

        if best_val <= CONV_THRESHOLD:
            converged = True
            gens_to_converge = gen
            terminal_state = "converged"
            break

        if gens_no_improve >= _plateau:
            terminal_state = "early_stopped"
            break

        # ---- reproduction (verbatim from deduce_formula) -------------------
        new_pop = [best_tree.copy()]
        while len(new_pop) < pop_size:
            t1 = evo_rng.sample(fitness_scores, k=5)
            t1.sort(key=lambda x: x[1])
            parent1 = t1[0][0]

            if not mutation_only and evo_rng.random() < 0.75 and len(new_pop) < pop_size - 1:
                t2 = evo_rng.sample(fitness_scores, k=5)
                t2.sort(key=lambda x: x[1])
                parent2 = t2[0][0]
                child1, child2 = crossover(parent1, parent2, max_depth=None)
                if count_nodes(child1) > MAX_NODES:
                    child1 = parent1.copy()
                if count_nodes(child2) > MAX_NODES:
                    child2 = parent2.copy()
                new_pop.extend([child1, child2])
            else:
                new_pop.append(parent1.copy())

        for i in range(1, len(new_pop)):
            if mutation_only:
                if use_point_mutate and evo_rng.random() < 0.05:
                    new_pop[i] = point_mutate(new_pop[i], OPERATORS, TERMINALS, VARIABLES)
                else:
                    cand = mutate(new_pop[i], OPERATORS, TERMINALS, VARIABLES,
                                  mut_depth, absolute_max_depth=MAX_DEPTH)
                    new_pop[i] = cand if count_nodes(cand) <= MAX_NODES else new_pop[i]
            else:
                if evo_rng.random() < 0.20:
                    cand = mutate(new_pop[i], OPERATORS, TERMINALS, VARIABLES,
                                  mut_depth, absolute_max_depth=MAX_DEPTH)
                    new_pop[i] = cand if count_nodes(cand) <= MAX_NODES else new_pop[i]
                elif use_point_mutate and evo_rng.random() < 0.05:
                    new_pop[i] = point_mutate(new_pop[i], OPERATORS, TERMINALS, VARIABLES)

        population = new_pop[:pop_size]
    else:
        terminal_state = "budget_exhausted"
        gens_run = generations

    if best_tree is None:
        best_tree = population[0]
        best_val = nmse(best_tree, val_xs, val_target)

    exact_match = False
    if grid_xs is not None:
        pred = vec_eval(best_tree, grid_xs)
        exact_match = (
            scaled_func_hash(pred, grid_scale) == scaled_func_hash(grid_target, grid_scale)
        )

    return {
        "converged": converged,
        "exact_hash_match": exact_match,
        "gens_to_converge": gens_to_converge,
        "final_nmse": best_val,
        "final_tree_size": count_nodes(best_tree),
        "terminal_state": terminal_state,
        "wall_seconds": time.time() - t0,
        "cpu_seconds": time.process_time() - c0,
        "gens_run": gens_run,
        "best_expr": expr_str(best_tree) if count_nodes(best_tree) <= 200 else "",
    }


# --------------------------------------------------------------------------
# Evaluator equivalence check (numpy path vs. the reference cmath path)
# --------------------------------------------------------------------------
def check_evaluator_agreement(n_trees=20000, seed=12345, n_points=32):
    """Compare vectorised fitness with ga-lib's per-point cmath evaluation on
    randomly generated trees drawn from the same generator the GP uses."""
    rng = random.Random(seed)
    import random as _random_module
    _random_module.choice = rng.choice
    _random_module.random = rng.random
    _random_module.sample = rng.sample
    _random_module.uniform = rng.uniform

    from generator import generate_random_tree

    xs_list = [rng.uniform(DOMAIN_LO, DOMAIN_HI) for _ in range(n_points)]
    xs = np.array([complex(v) for v in xs_list], dtype=np.complex128)
    tgt_tree = parse_expr("eml(x, eml(eml(x, x), x))")
    tgt = vec_eval(tgt_tree, xs)
    tgt_list = [complex(v) for v in tgt]

    def ref_mse(tree):
        error = 0.0
        try:
            for xv, tv in zip(xs_list, tgt_list):
                pred = tree.evaluate(x=xv)
                diff = abs(pred - tv)
                if math.isnan(diff) or math.isinf(diff):
                    error += 1000.0
                else:
                    error += diff ** 2
            return error / len(xs_list)
        except Exception:
            return _INF

    n_mismatch = 0
    worst_rel = 0.0
    n_finite = 0
    for _ in range(n_trees):
        depth = rng.randint(1, MAX_DEPTH)
        method = "full" if rng.random() < 0.5 else "grow"
        tree = generate_random_tree(depth, OPERATORS, TERMINALS, VARIABLES, method)
        if count_nodes(tree) > MAX_NODES:
            continue
        a = mse_against(vec_eval(tree, xs), tgt)
        b = ref_mse(tree)
        if math.isinf(a) or math.isinf(b) or a != a or b != b:
            if not (math.isinf(a) and math.isinf(b)):
                n_mismatch += 1
            continue
        n_finite += 1
        denom = max(abs(a), abs(b), 1e-300)
        rel = abs(a - b) / denom
        worst_rel = max(worst_rel, rel)
        if rel > 1e-9:
            n_mismatch += 1
    return {
        "n_trees_compared": n_finite,
        "n_fitness_mismatches": n_mismatch,
        "worst_relative_fitness_difference": worst_rel,
    }

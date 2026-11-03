import sys
import os
import argparse
import random
import math
import cmath

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ga-lib')))
from generator import generate_ramped_population
from evolution import crossover, mutate, point_mutate
from node import Node
from test_suite import TEST_MATRIX
from eml_simplify import simplify_eml_tree

# ---------------------------------------------------------------------------
# Shared budget — both modes use identical limits so any difference in
# outcome reflects representation power, not search advantage.
#
# MAX_NODES drives max_depth: a full binary tree of depth d has 2^(d+1)-1
# nodes, so max_depth = ceil(log2(MAX_NODES+1)) - 1 = 13 for 10_000 nodes.
# This means EML can grow trees large enough to represent sin (K=351,
# depth≥8) and cos (K=269), which a hard-coded depth-6 cap prevented.
# ---------------------------------------------------------------------------
MAX_NODES       = 2_000
MAX_DEPTH       = math.ceil(math.log2(MAX_NODES + 1)) - 1   # 10
BATCH_SIZE      = 75
VALIDATION_SIZE = 200
# plateau_gens = max(50, generations // 10) computed per run


def count_nodes(tree):
    if tree is None:
        return 0
    if not tree.is_operator:
        return 1
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)


def main():
    parser = argparse.ArgumentParser(description="Deduce formula using standard or EML GP")
    parser.add_argument("--pop_size",    type=int,  default=500)
    parser.add_argument("--generations", type=int,  default=100)
    parser.add_argument("--use_eml",     action="store_true")
    parser.add_argument("--test_id",     type=str,  default=None)
    parser.add_argument("--seed",          type=int,  default=None,
                        help="Master RNG seed. Derives separate sub-seeds for "
                             "validation set, population init, and evolution.")
    parser.add_argument("--plateau_mult", type=int,  default=1,
                        help="Multiply plateau patience by this factor. "
                             "Use 3 for EML to account for its slower structural convergence.")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="How often to print generation metrics")
    parser.add_argument("--mutation_only", action="store_true",
                        help="Disable crossover, use 100% mutation (95% subtree, 5% point).")
    parser.add_argument("--std_ops", type=str, default=None,
                        help="Comma-separated override for the standard-GP operator set, e.g. "
                             "'+,-,*,/,exp,log'. Fixed-primitive-set control (R1.3). "
                             "Default: the per-target set from TEST_MATRIX (unchanged behaviour).")
    # Rerun flags (2026-09 revision). Defaults reproduce the original runs exactly.
    parser.add_argument("--no_size_penalty", action="store_true",
                        help="Plain MSE fitness: drop the 1e-4*nodes term applied below MSE 1e-5.")
    parser.add_argument("--trace", action="store_true",
                        help="Compact per-generation log (full precision, no formulas) plus FINAL_* summary lines.")
    parser.add_argument("--no_simplify", action="store_true",
                        help="Skip the post-run sympy simplification of EML trees.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Three isolated Random instances derived from the master seed.
    # This guarantees:
    #   • val_rng  — validation set is identical for both modes (fair eval)
    #   • pop_rng  — initial population is independently seeded per mode
    #   • evo_rng  — evolution decisions are independently seeded per mode
    # Using random.Random() objects instead of the global random module
    # so the three streams never interfere with each other and are safe
    # to call from any context.
    # ------------------------------------------------------------------
    master = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    _m = random.Random(master)
    val_seed = _m.randint(0, 2**31 - 1)
    pop_seed = _m.randint(0, 2**31 - 1)
    evo_seed = _m.randint(0, 2**31 - 1)

    val_rng = random.Random(val_seed)
    pop_rng = random.Random(pop_seed)
    evo_rng = random.Random(evo_seed)

    # Monkey-patch the module-level random used by generator/evolution.
    # Both modules call `random.choice`, `random.random`, `random.sample`,
    # `random.uniform` — we redirect them to evo_rng so evolution is
    # reproducible and isolated from the validation-set draws.
    import random as _random_module
    _random_module.choice  = evo_rng.choice
    _random_module.random  = evo_rng.random
    _random_module.sample  = evo_rng.sample
    _random_module.uniform = evo_rng.uniform
    _random_module.seed    = lambda *a, **kw: None   # block accidental re-seeding

    # ------------------------------------------------------------------
    # Test configuration
    # ------------------------------------------------------------------
    variables   = ['x', 'y']
    data_range  = (-10.0, 10.0)
    target_func = lambda x, y: (x * y) - 2.0
    std_ops     = ['+', '-', '*', '/']

    if args.test_id and args.test_id in TEST_MATRIX:
        test        = TEST_MATRIX[args.test_id]
        variables   = test["vars"]
        data_range  = test["range"]
        target_func = test["func"]
        std_ops     = test["std_ops"]
        print(f"Loaded Test {args.test_id}: {test['name']}")

    if args.std_ops:
        std_ops = [op.strip() for op in args.std_ops.split(",") if op.strip()]
        print(f"Operator-set override active: {std_ops}")

    terminals = [1.0]

    if args.use_eml:
        operators = ['eml']
        print("Mode: EML-only  |  operator set: ['eml']  |  terminal set: [1.0]")
    else:
        operators = std_ops
        print(f"Mode: Standard  |  operator set: {operators}  |  terminal set: [1.0]")

    plateau_gens = max(50, args.generations // 10) * args.plateau_mult
    print(f"Budget: MAX_DEPTH={MAX_DEPTH}, MAX_NODES={MAX_NODES}, "
          f"BATCH={BATCH_SIZE}, VAL={VALIDATION_SIZE}, "
          f"PLATEAU_STOP={plateau_gens}  master_seed={master}")

    # ------------------------------------------------------------------
    # Fixed held-out validation set — drawn from val_rng, identical for
    # both EML and standard modes when master seed is the same.
    # ------------------------------------------------------------------
    val_inputs = [
        [val_rng.uniform(data_range[0], data_range[1]) for _ in variables]
        for _ in range(VALIDATION_SIZE)
    ]

    # Population init uses pop_rng via the monkey-patched random module.
    # Temporarily redirect to pop_rng for init, then back to evo_rng.
    _random_module.choice  = pop_rng.choice
    _random_module.random  = pop_rng.random
    _random_module.sample  = pop_rng.sample
    _random_module.uniform = pop_rng.uniform

    population = generate_ramped_population(
        args.pop_size, MAX_DEPTH, operators, terminals, variables
    )

    # Switch to evo_rng for all evolution decisions.
    _random_module.choice  = evo_rng.choice
    _random_module.random  = evo_rng.random
    _random_module.sample  = evo_rng.sample
    _random_module.uniform = evo_rng.uniform

    # ------------------------------------------------------------------
    # Fitness helpers
    # ------------------------------------------------------------------
    def _mse(tree, inputs):
        if count_nodes(tree) > MAX_NODES:
            return float('inf')
        error      = 0.0
        valid_eval = 0
        try:
            for inp in inputs:
                kwargs   = {var: val for var, val in zip(variables, inp)}
                pred     = tree.evaluate(**kwargs)
                expected = target_func(*inp)
                diff     = abs(pred - expected)
                if cmath.isnan(diff) or cmath.isinf(diff):
                    error += 1000.0
                else:
                    error += diff ** 2
                valid_eval += 1
            return error / valid_eval if valid_eval > 0 else float('inf')
        except Exception:
            return float('inf')

    def training_fitness(tree, batch):
        mse = _mse(tree, batch)
        if not args.no_size_penalty and mse < 1e-5:
            return mse + count_nodes(tree) * 0.0001
        return mse

    def validation_error(tree):
        return _mse(tree, val_inputs)

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    # Mutation subtree depth: sqrt of max depth, minimum 2.
    # Scales with MAX_DEPTH so EML can grow large subtrees via mutation.
    mut_depth   = max(2, round(math.sqrt(MAX_DEPTH)))   # sqrt(13) ≈ 4

    best_tree       = None
    best_val_error  = float('inf')
    gens_no_improve = 0
    converged       = False

    # EML with a single operator gets no diversity from point_mutate
    # (eml→eml is always the only choice). Skip it to avoid wasting
    # 5% of reproductive budget on a structural no-op.
    use_point_mutate = len(operators) > 1 or not all(op == 'eml' for op in operators)
    stop_reason = "budget"

    for gen in range(args.generations):
        batch = [
            [evo_rng.uniform(data_range[0], data_range[1]) for _ in variables]
            for _ in range(BATCH_SIZE)
        ]

        fitness_scores = [
            (ind, training_fitness(ind, batch)) for ind in population
        ]
        fitness_scores.sort(key=lambda item: item[1])

        gen_best_tree = fitness_scores[0][0]
        gen_val_err   = validation_error(gen_best_tree)
        n             = count_nodes(gen_best_tree)

        if gen_val_err < best_val_error:
            best_val_error  = gen_val_err
            best_tree       = gen_best_tree
            gens_no_improve = 0
        else:
            gens_no_improve += 1

        if args.trace:
            _sizes = sorted(count_nodes(ind) for ind in population)
            print(f"GEN {gen} BEST {best_val_error:.9e} VAL {gen_val_err:.9e} "
                  f"NODES {n} MEDIAN {_sizes[len(_sizes)//2]}", flush=True)
        elif gen % args.print_interval == 0 or gen == args.generations - 1:
            tree_sizes = sorted([count_nodes(ind) for ind in population])
            median_nodes = tree_sizes[len(tree_sizes)//2]
            print(
                f"Generation {gen:03d}"
                f" - Best Error: {best_val_error:.6f}"
                f" - Val Error: {gen_val_err:.6f}"
                f" - Nodes: {n:>5}"
                f" - Median Nodes: {median_nodes:>5}"
                f" - Formula: {gen_best_tree.to_formula()}"
            )

        if best_val_error <= 1e-6:
            print(f"Perfect solution found at generation {gen}! Stopping.")
            converged = True
            stop_reason = "converged"
            break

        if gens_no_improve >= plateau_gens:
            print(f"Plateau detected ({plateau_gens} gens without improvement). Stopping at gen {gen}.")
            stop_reason = "plateau"
            break

        # ------------------------------------------------------------------
        # Reproduction: tournament (k=5), crossover, mutation.
        # ------------------------------------------------------------------
        new_pop = [best_tree.copy()]  # elitism

        while len(new_pop) < args.pop_size:
            t1 = evo_rng.sample(fitness_scores, k=5)
            t1.sort(key=lambda x: x[1])
            parent1 = t1[0][0]

            if not args.mutation_only and evo_rng.random() < 0.75 and len(new_pop) < args.pop_size - 1:
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
            if args.mutation_only:
                # 100% mutation: 95% subtree, 5% point
                if use_point_mutate and evo_rng.random() < 0.05:
                    new_pop[i] = point_mutate(new_pop[i], operators, terminals, variables)
                else:
                    candidate = mutate(
                        new_pop[i], operators, terminals, variables,
                        mut_depth, absolute_max_depth=MAX_DEPTH
                    )
                    new_pop[i] = candidate if count_nodes(candidate) <= MAX_NODES else new_pop[i]
            else:
                if evo_rng.random() < 0.20:
                    candidate = mutate(
                        new_pop[i], operators, terminals, variables,
                        mut_depth, absolute_max_depth=MAX_DEPTH
                    )
                    new_pop[i] = candidate if count_nodes(candidate) <= MAX_NODES else new_pop[i]
                elif use_point_mutate and evo_rng.random() < 0.05:
                    new_pop[i] = point_mutate(new_pop[i], operators, terminals, variables)

        population = new_pop[:args.pop_size]

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    if best_tree is None:
        best_tree      = population[0]
        best_val_error = validation_error(best_tree)

    n           = count_nodes(best_tree)
    raw_formula = best_tree.to_formula()
    readable    = simplify_eml_tree(best_tree) if (args.use_eml and not args.no_simplify) else raw_formula
    if args.trace:
        print(f"STOP_REASON {stop_reason}")
        print(f"LAST_GEN {gen}")
        print(f"FINAL_ERR {best_val_error!r}")
        print(f"FINAL_NODES {n}")
        print(f"FINAL_TREE {raw_formula}", flush=True)
    mode_label  = "EML" if args.use_eml else "Standard"

    print("\n===============================")
    print(f"{mode_label} GP Evolution complete.")
    print(f"Best Formula: Result = {readable}")
    if args.use_eml:
        print(f"EML Raw Tree: {raw_formula}")
    print(f"Node count: {n}")
    print(f"Final MSE:  {best_val_error:.8f}")
    if converged:
        print("Status: CONVERGED (perfect solution)")
    else:
        print("Status: BUDGET EXHAUSTED")


if __name__ == "__main__":
    main()

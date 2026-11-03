import random
from node import Node

# Operators that take only one child (left); all others are binary.
UNARY_OPS = {'sin', 'cos', 'exp', 'log', 'tan', 'sinh', 'cosh', 'tanh', 'atan', 'sqrt'}

def _is_unary(op):
    return op in UNARY_OPS

def _make_leaf(variables, terminals):
    # Weight leaf choices so each variable and each constant is equally likely.
    # With n variables and m terminals each slot has probability 1/(n+m).
    # This prevents variables from being crowded out when n > 1.
    choices = list(variables) + list(terminals)
    val = random.choice(choices)
    is_var = isinstance(val, str)
    return Node(val, is_operator=False)

def generate_tree_full(max_depth, operators, terminals, variables=None, current_depth=0):
    if variables is None:
        variables = []

    if current_depth >= max_depth:
        return _make_leaf(variables, terminals)

    op = random.choice(operators)
    node = Node(op, is_operator=True)
    node.left = generate_tree_full(max_depth, operators, terminals, variables, current_depth + 1)
    if not _is_unary(op):
        node.right = generate_tree_full(max_depth, operators, terminals, variables, current_depth + 1)
    return node

def generate_tree_grow(max_depth, operators, terminals, variables=None, current_depth=0):
    if variables is None:
        variables = []

    if current_depth >= max_depth:
        return _make_leaf(variables, terminals)

    # At non-root positions, allow early termination with 20% probability.
    if current_depth > 0 and random.random() < 0.2:
        return _make_leaf(variables, terminals)

    op = random.choice(operators)
    node = Node(op, is_operator=True)
    node.left = generate_tree_grow(max_depth, operators, terminals, variables, current_depth + 1)
    if not _is_unary(op):
        node.right = generate_tree_grow(max_depth, operators, terminals, variables, current_depth + 1)
    return node

def generate_random_tree(max_depth, operators, terminals, variables=None, method=None):
    """
    Method can be 'full', 'grow', or None (where it defaults to grow for backwards compatibility
    if not using a ramped half-and-half population generator).
    """
    if method == 'full':
        return generate_tree_full(max_depth, operators, terminals, variables, 0)
    else:
        return generate_tree_grow(max_depth, operators, terminals, variables, 0)

def generate_ramped_population(pop_size, max_depth, operators, terminals, variables=None):
    """
    Ramped half-and-half initialization.

    Guarantees each variable appears as the root leaf in at least one tree in
    the initial population so no variable is structurally absent at generation 0.
    """
    variables = variables or []
    population = []

    # Seed: one depth-1 tree per variable that is just that variable as a leaf,
    # ensuring every input dimension is represented from the start.
    for var in variables:
        if len(population) < pop_size:
            leaf = Node(var, is_operator=False)
            population.append(leaf)

    depths = list(range(2, max_depth + 1)) if max_depth >= 2 else [max_depth]
    pop_per_depth = max(1, pop_size // len(depths))

    for depth in depths:
        for i in range(pop_per_depth):
            if len(population) >= pop_size:
                break
            method = 'full' if i % 2 == 0 else 'grow'
            population.append(generate_random_tree(depth, operators, terminals, variables, method))

    while len(population) < pop_size:
        method = random.choice(['full', 'grow'])
        population.append(generate_random_tree(max_depth, operators, terminals, variables, method))

    return population

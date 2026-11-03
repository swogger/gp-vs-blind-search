import random
from node import Node
from generator import generate_random_tree, UNARY_OPS

def get_depth(tree):
    if tree is None:
        return 0
    return 1 + max(get_depth(tree.left), get_depth(tree.right))

def get_random_node(tree):
    """Returns a random node from the tree and its parent."""
    nodes = []
    def traverse(node, parent=None, is_left=None):
        if node is not None:
            nodes.append((node, parent, is_left))
            traverse(node.left, node, True)
            traverse(node.right, node, False)
    traverse(tree)
    return random.choice(nodes) if nodes else (None, None, None)

def crossover(tree1, tree2, max_depth=15):
    """Swaps random subtrees between two trees."""
    new_tree1 = tree1.copy()
    new_tree2 = tree2.copy()
    
    node1, parent1, is_left1 = get_random_node(new_tree1)
    node2, parent2, is_left2 = get_random_node(new_tree2)
    
    if node1 is None or node2 is None:
        return new_tree1, new_tree2
        
    if parent1 is None:
        new_tree1 = node2.copy()
    elif is_left1:
        parent1.left = node2.copy()
    else:
        parent1.right = node2.copy()
        
    if parent2 is None:
        new_tree2 = node1.copy()
    elif is_left2:
        parent2.left = node1.copy()
    else:
        parent2.right = node1.copy()
        
    if max_depth is not None:
        if get_depth(new_tree1) > max_depth:
            new_tree1 = tree1.copy()
        if get_depth(new_tree2) > max_depth:
            new_tree2 = tree2.copy()
            
    return new_tree1, new_tree2

def mutate(tree, operators, terminals, variables=None, max_depth=2, absolute_max_depth=15):
    """Replaces a random subtree with a new random tree."""
    new_tree = tree.copy()
    node, parent, is_left = get_random_node(new_tree)
    
    if node is None:
        new_subtree = generate_random_tree(max_depth, operators, terminals, variables)
        return new_subtree if get_depth(new_subtree) <= absolute_max_depth else tree
        
    new_subtree = generate_random_tree(max_depth, operators, terminals, variables)
    
    if parent is None:
        candidate = new_subtree
    elif is_left:
        parent.left = new_subtree
        candidate = new_tree
    else:
        parent.right = new_subtree
        candidate = new_tree
        
    if get_depth(candidate) > absolute_max_depth:
        return tree
    return candidate

def point_mutate(tree, operators, terminals, variables=None):
    """Selects a random node and changes its value/operator without altering tree structure."""
    new_tree = tree.copy()
    node, parent, is_left = get_random_node(new_tree)
    
    if node is None:
        return new_tree
        
    if node.is_operator:
        # Preserve arity: unary ops stay unary, binary ops stay binary.
        if node.value in UNARY_OPS:
            available_ops = [op for op in operators if op in UNARY_OPS]
        else:
            available_ops = [op for op in operators if op not in UNARY_OPS]
            
        if available_ops:
            node.value = random.choice(available_ops)
    else:
        # Swap terminal/variable — equal weight across all variables and constants,
        # consistent with _make_leaf in generator.py.
        choices = list(variables or []) + list(terminals)
        node.value = random.choice(choices)
        
    return new_tree

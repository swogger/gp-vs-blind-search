import random
import sys
import os

# Ensure we can import the modules from the current directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from population import initialize_population, evolve

# Target function: x^2 + 2x + 1
def target_function(x):
    return x**2 + 2*x + 1

# Fitness function: mean squared error over a set of points
def fitness_function(tree):
    points = [-2, -1, 0, 1, 2]
    error = 0
    try:
        for x in points:
            pred = tree.evaluate(x=x)
            actual = target_function(x)
            error += (pred - actual) ** 2
        return error / len(points)
    except Exception as e:
        # Penalize invalid evaluations (e.g., division by zero resulting in very large numbers, though handled safely in our Node)
        return float('inf')

def main():
    operators = ['+', '-', '*', '/']
    terminals = [1, 2, 3] # Constants
    variables = ['x']     # Variables
    
    pop_size = 100
    generations = 20
    max_depth = 3
    
    print("Initializing population...")
    population = initialize_population(pop_size, max_depth, operators, terminals, variables)
    
    print("Evolving...")
    best_tree, final_population = evolve(
        population=population,
        fitness_function=fitness_function,
        generations=generations,
        operators=operators,
        terminals=terminals,
        variables=variables,
        mutation_rate=0.2,
        crossover_rate=0.7,
        max_depth=4
    )
    
    print("\nEvolution complete.")
    print(f"Best Formula: {best_tree.to_formula()}")
    print("Evaluating Best Formula on test points:")
    for x in [-3, 0, 3]:
        print(f"  f({x}) = {best_tree.evaluate(x=x)} (Target: {target_function(x)})")

if __name__ == "__main__":
    # Seed for reproducibility in this example
    random.seed(42)
    main()

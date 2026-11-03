import random
from generator import generate_random_tree
from evolution import crossover, mutate

def initialize_population(size, max_depth, operators, terminals, variables=None):
    return [generate_random_tree(max_depth, operators, terminals, variables) for _ in range(size)]

def evolve(population, fitness_function, generations, operators, terminals, variables=None, 
           mutation_rate=0.1, crossover_rate=0.8, max_depth=4, elitism=True):
    current_pop = population
    
    for gen in range(generations):
        # Calculate fitness for all individuals
        fitness_scores = [(ind, fitness_function(ind)) for ind in current_pop]
        # Sort by fitness (assuming lower is better, i.e., minimizing error)
        fitness_scores.sort(key=lambda x: x[1])
        
        print(f"Generation {gen} - Best Fitness (Error): {fitness_scores[0][1]:.4f} - Best Formula: {fitness_scores[0][0].to_formula()}")
        
        # If perfect fitness is found
        if fitness_scores[0][1] <= 1e-6:
            print("Perfect solution found!")
            return fitness_scores[0][0], current_pop
            
        new_pop = []
        if elitism:
            # Keep the best individual
            new_pop.append(fitness_scores[0][0].copy())
            
        while len(new_pop) < len(population):
            # Tournament selection
            tournament = random.sample(fitness_scores, k=min(3, len(fitness_scores)))
            tournament.sort(key=lambda x: x[1])
            parent1 = tournament[0][0]
            
            if random.random() < crossover_rate and len(new_pop) < len(population) - 1:
                tournament2 = random.sample(fitness_scores, k=min(3, len(fitness_scores)))
                tournament2.sort(key=lambda x: x[1])
                parent2 = tournament2[0][0]
                
                child1, child2 = crossover(parent1, parent2)
                new_pop.extend([child1, child2])
            else:
                new_pop.append(parent1.copy())
                
        # Apply mutation
        for i in range(1 if elitism else 0, len(new_pop)):
            if random.random() < mutation_rate:
                new_pop[i] = mutate(new_pop[i], operators, terminals, variables, max_depth)
                
        current_pop = new_pop[:len(population)]
        
    # Final evaluation to return the best
    fitness_scores = [(ind, fitness_function(ind)) for ind in current_pop]
    fitness_scores.sort(key=lambda x: x[1])
    return fitness_scores[0][0], current_pop

from genetic_operations import *

def run_genetic_algorithm(total_gpus, total_vram, total_types, total_prns, prns, params):
    """
    Generates a solution for the OGD problem instance using a genetic algorithm.
    """
    # Extract all algorithm cmd parameters
    population_size = params.population_size
    crossover_rate = params.crossover_rate
    mutation_rate = params.mutation_rate
    elitism_rate = params.elitism_rate
    selection_pressure = params.selection_pressure
    max_iterations = params.max_iterations
    stagnation_limit = params.stagnation_limit
    seed = params.seed
    total_elites = int(population_size * elitism_rate)

    # Set random seed for reproducibility
    if(seed):
        np.random.seed(seed)

    # Create initial population
    population, gpu_vram_population, gpu_type_dist_population, fitness_population, best_solution = generate_initial_population(
        population_size, total_gpus, total_vram, total_types, total_prns, prns
    )

    i = 0
    stagnation_counter = 0
    while (i < max_iterations) and (stagnation_counter < stagnation_limit):
        new_population = np.empty((population_size, total_prns), dtype=int)                               # Valid generated solutions/chromosomes
        new_gpu_vram_population = np.empty((population_size, total_gpus), dtype=int)                      # GPUs remaining VRAM per solution/chromosome
        new_gpu_type_dist_population = np.empty((population_size, total_gpus, total_types), dtype=int)    # GPUs type distribution per solution/chromosome
        new_fitness_population = np.empty(population_size, dtype=int)                                     # Fitness per solution/chromosome
    
        stagnated = True

        # ELITISM
        elite_indices = np.argsort(fitness_population)

        best_solution = elite_indices[0]
        for j in range(total_elites):
            new_population[j] = population[elite_indices[j]]
            new_gpu_vram_population[j] = gpu_vram_population[elite_indices[j]]
            new_gpu_type_dist_population[j] = gpu_type_dist_population[elite_indices[j]]
            new_fitness_population[j] = fitness_population[elite_indices[j]]

        for k in range(total_elites, population_size):            
            # SELECT PARENTS
            parent1_index, parent2_index = select_parents(fitness_population, selection_pressure, population_size)
            parent1, parent2 = population[parent1_index], population[parent2_index]

            # CROSSOVER
            if rand.random() <= crossover_rate:
                solution, gpu_vram, gpu_type_dist, fitness = crossover_solutions(
                    parent1, parent2, prns, total_prns, total_types, total_gpus, total_vram
                )
            else:
                solution = parent1
                gpu_vram = gpu_vram_population[parent1_index]
                gpu_type_dist = gpu_type_dist_population[parent1_index]
                fitness = fitness_population[parent1_index]

            # MUTATION
            if rand.random() <= mutation_rate:
                solution, gpu_vram, gpu_type_dist, fitness = mutate_solution(
                    solution, gpu_vram, gpu_type_dist, fitness, prns, total_prns, total_gpus
                )

            # Add generated solution to new population
            new_population[k] = solution
            new_gpu_vram_population[k] = gpu_vram
            new_gpu_type_dist_population[k] = gpu_type_dist
            new_fitness_population[k] = fitness 

            # Update best solution
            if fitness < new_fitness_population[best_solution]: 
                best_solution = k
                stagnated = False
        
        # Update stagnation counter
        if stagnated:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        # Update old population with new population
        population = np.copy(new_population)
        gpu_vram_population = np.copy(new_gpu_vram_population)
        gpu_type_dist_population = np.copy(new_gpu_type_dist_population)
        fitness_population = np.copy(new_fitness_population)

        i += 1

    print(population[best_solution])
    print(gpu_vram_population[best_solution])
    print(gpu_type_dist_population[best_solution])
    print(fitness_population[best_solution])
    print(stagnated)
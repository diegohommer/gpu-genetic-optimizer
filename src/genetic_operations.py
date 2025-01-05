import random as rand
import numpy as np

def generate_feasible_solution(prns: int, total_prns: int, total_types: int, total_gpus: int, total_vram: int):
    """
    Finds a feasible solution for the OGD problem instance.
    """
    # Sorts the PRNs first by type and then by VRAM (descending)
    prns_sorted_by_type_and_vram = sorted(prns, key=lambda prn: (prn['prn_type'], -prn['prn_vram']))

    gpu_vram = np.full(total_gpus, total_vram, dtype=int)           # Remaining VRAM per GPU
    gpu_type_dist = np.zeros((total_gpus, total_types), dtype=int)  # Type distribution (GPUs x Types)
    solution = np.full(total_prns, -1, dtype=int)                   # Solution array (-1 => unallocated) 
    fitness = 0

    # First Fit Descending (FFD) greedy heuristic to try to find a valid solution
    for prn_index, prn in enumerate(prns_sorted_by_type_and_vram):  # Usando enumerate para pegar o índice
        prn_vram = prn['prn_vram']
        prn_type = prn['prn_type'] 

        # Check each GPU for sufficient VRAM        
        for gpu_index in range(total_gpus):            
            if prn_vram <= gpu_vram[gpu_index]: 
                solution[prn_index] = gpu_index                   
                gpu_vram[gpu_index] -= prn_vram              # Subtract VRAM from GPU
                gpu_type_dist[gpu_index][prn_type] += 1      # Increment type count for this GPU
                if gpu_type_dist[gpu_index][prn_type] == 1:
                    fitness += 1  # Increment fitness when the first PRN of this type is placed
                break
        else:
           # If heuristic fails to find a valid solution run for the hills!
            return False

    return solution, gpu_vram, gpu_type_dist, fitness


def generate_initial_population(population_size: int, total_gpus: int, total_vram: int, total_types: int, total_prns: int, prns: np.ndarray):  
    """
    Generate initial solution population for the OGD problem instance
    """
    population = np.empty((population_size, total_prns), dtype=int)                               # Valid generated solutions/chromosomes
    gpu_vram_population = np.empty((population_size, total_gpus), dtype=int)                      # GPUs remaining VRAM per solution/chromosome
    gpu_type_dist_population = np.empty((population_size, total_gpus, total_types), dtype=int)    # GPUs type distribution per solution/chromosome
    fitness_population = np.empty(population_size, dtype=int)                                     # Fitness per solution/chromosome
    best_solution = 0                                                                             # Best solution population index

    # Insert valid solution data into populations
    valid_solution, solution_vram, solution_type_dist, solution_fitness = generate_feasible_solution(prns, total_prns, total_types, total_gpus, total_vram)
    population[0] = valid_solution
    gpu_vram_population[0] = solution_vram
    gpu_type_dist_population[0] = solution_type_dist
    fitness_population[0] = solution_fitness

    # Mutate valid solution to generate the rest of the population
    for i in range(1,population_size):
        population[i], gpu_vram_population[i], gpu_type_dist_population[i], fitness_population[i] = mutate_solution(
            solution=valid_solution,
            gpu_vram=solution_vram,
            gpu_type_dist=solution_type_dist,
            fitness=solution_fitness,
            prns=prns,
            total_prns=total_prns,
            total_gpus=total_gpus
        )
        if(fitness_population[i] < fitness_population[best_solution]): 
            best_solution = i 

    return population, gpu_vram_population, gpu_type_dist_population, fitness_population, best_solution


def mutate_solution(solution, gpu_vram, gpu_type_dist, fitness: int, prns, total_prns: int, total_gpus: int):
    """
    Randomly mutates the input chromosome (solution) by changing the allocation of PRNs
    """
    mutated_solution = solution.copy()
    mutated_gpu_vram = gpu_vram.copy()
    mutated_gpu_type_dist = gpu_type_dist.copy()

    # Sort a random PRN to be reallocated
    prn_index = rand.randint(0, total_prns-1)
    prn_vram = prns[prn_index]['prn_vram']
    prn_type = prns[prn_index]['prn_type']
    old_gpu_index = mutated_solution[prn_index]

    # Sort a valid GPU for the PRN to be reallocated to
    valid_gpus = [i for i in range(total_gpus) if mutated_gpu_vram[i] >= prn_vram and i != old_gpu_index]
    if valid_gpus:  
        new_gpu_index = rand.choice(valid_gpus)

        # Update VRAM & type distribution of old and new GPU
        mutated_gpu_vram[old_gpu_index] += prn_vram           
        mutated_gpu_type_dist[old_gpu_index][prn_type] -= 1
        mutated_gpu_vram[new_gpu_index] -= prn_vram 
        mutated_gpu_type_dist[new_gpu_index][prn_type] += 1

        # Update fitness of mutated solution
        if mutated_gpu_type_dist[old_gpu_index][prn_type] == 0: 
            fitness -= 1
        if mutated_gpu_type_dist[new_gpu_index][prn_type] == 1: 
            fitness += 1

        mutated_solution[prn_index] = new_gpu_index

    return mutated_solution, mutated_gpu_vram, mutated_gpu_type_dist, fitness
    

def crossover_solutions(parent1, parent2, prns, total_prns, total_types, total_gpus, total_vram):
    """ Combines two parental solutions into a new child solution, prioritizing the minimization
    of type dispersion of PRNs across GPUs and attempting to group similar PRNs on the same GPU. """

    # Initializes solution, remaining VRAM, and type distribution
    child_solution = np.full(total_prns, -1, dtype=int)
    child_gpu_vram = np.full(total_gpus, total_vram, dtype=int)
    child_gpu_type_dist = np.zeros((total_gpus, total_types), dtype=int)
    child_fitness = 0

    # Iterates over each PRN
    for prn_index in range(total_prns):
        prn_vram = prns[prn_index]['prn_vram']
        prn_type = prns[prn_index]['prn_type']
        
        # Selects a GPU candidate from the parents
        candidate_gpu = None

        # Attempts to inherit the allocation from the parents, prioritizing the highest number of PRNs of the same type.
        for parent in (parent1, parent2):
            gpu = parent[prn_index]
            if gpu != -1 and child_gpu_vram[gpu] >= prn_vram:
                # Prioritizes the GPU that already has the most PRNs of the same type.
                if candidate_gpu is None or child_gpu_type_dist[gpu][prn_type] > child_gpu_type_dist[candidate_gpu][prn_type]:
                    candidate_gpu = gpu
        
        # If no valid GPU is inherited, selects a new GPU with the least type dispersion.
        if candidate_gpu is None:
            valid_gpus = [
                gpu for gpu in range(total_gpus)
                if child_gpu_vram[gpu] >= prn_vram
            ]
            if valid_gpus:
                # For the new GPU, prioritizes selecting one that already has PRNs of the same type.
                candidate_gpu = min(
                    valid_gpus,
                    key=lambda g: (child_gpu_type_dist[g][prn_type], child_gpu_vram[g])
                )
            else:
                # If no valid GPU is found, attempts to generate a feasible solution.
                return generate_feasible_solution(prns, total_prns, total_types, total_gpus, total_vram)

        # Updates the allocation for the child.
        child_solution[prn_index] = candidate_gpu
        child_gpu_vram[candidate_gpu] -= prn_vram
        child_gpu_type_dist[candidate_gpu][prn_type] += 1

        # Updates the fitness
        if child_gpu_type_dist[candidate_gpu][prn_type] == 1:
            child_fitness += 1

    return child_solution, child_gpu_vram, child_gpu_type_dist, child_fitness


def select_parents(fitness_population: np.ndarray, selection_pressure: float, population_size: int):
    """
    Select parents from the population using roulette wheel selection with a constant selection pressure.
    
    Args:
        fitness_population (numpy.ndarray): Array containing the fitness values of each individual in the population.
        selection_pressure (float): A constant value representing the selection pressure. Values > 1 increase pressure, values < 1 decrease pressure.
        
    Returns:
        tuple: Two selected parents indices.
    """
    # Scale fitness values based on the selection pressure
    scaled_fitness = fitness_population ** selection_pressure
    probabilities = scaled_fitness / scaled_fitness.sum()
    
    # Perform roulette wheel selection
    parent1 = np.random.choice(population_size, p=probabilities)
    parent2 = np.random.choice(population_size, p=probabilities)
    
    return parent1, parent2

def print_ga_outputs(population, gpu_vram_population, gpu_type_dist_population, fitness_population, best_solution, stagnated):
    """
    Prints the outputs of the genetic algorithm in a readable format.

    Args:
        population : numpy.ndarray
            The array representing the population of solutions (chromosomes).
        gpu_vram_population : numpy.ndarray
            The array representing the remaining VRAM per GPU for each solution in the population.
        gpu_type_dist_population : numpy.ndarray
            The array representing the type distribution of GPUs for each solution in the population.
        fitness_population : numpy.ndarray
            The array containing the fitness values for each solution in the population.
        best_solution : int
            The index of the best solution in the population.
        stagnated : bool
            Indicates whether the algorithm has stagnated.

    Returns:
        None (This function prints the results directly to the console.)
    """
    print("\n=== Genetic Algorithm Results ===")
    print(f"Best Solution Index: {best_solution}")
    print("\nBest Solution (Chromosome):")
    print(population[best_solution])
    
    print("\nRemaining VRAM per GPU for Best Solution:")
    print(gpu_vram_population[best_solution])
    
    print("\nType Distribution per GPU for Best Solution:")
    for gpu_idx, gpu_type_dist in enumerate(gpu_type_dist_population[best_solution]):
        print(f"GPU {gpu_idx + 1}: {gpu_type_dist}")
    
    print("\nFitness of Best Solution:")
    print(fitness_population[best_solution])
    
    print("\nAlgorithm Stagnated:")
    print("Yes" if stagnated else "No")
    print("=================================\n")

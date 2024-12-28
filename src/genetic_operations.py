import random as rand
import numpy as np


def generate_feasible_solution(prns: int, total_prns: int, total_types: int, total_gpus: int, total_vram: int):
    """
    Finds a feasible solution for the OGD problem instance.
    """
    prns_sorted_by_vram = np.argsort(-prns['prn_vram'])             # PRNs sorted by VRAM consumption (descending)
    gpu_vram = np.full(total_gpus, total_vram, dtype=int)           # Remaining VRAM per GPU
    gpu_type_dist = np.zeros((total_gpus, total_types), dtype=int)  # Type distribution (GPUs x Types)
    solution = np.full(total_prns, -1, dtype=int)                   # Solution array (-1 => unallocated) 
    fitness = 0

    # First Fit Descending (FFD) greedy heuristic to try to find a valid solution
    for prn_index in prns_sorted_by_vram:
        prn_vram = prns[prn_index]['prn_vram']
        prn_type = prns[prn_index]['prn_type'] 

        # Check each GPU for sufficient VRAM        
        for gpu_index in range(total_gpus):            
            if (prn_vram <= gpu_vram[gpu_index]): 
                solution[prn_index] = gpu_index                   
                gpu_vram[gpu_index] -= prn_vram             
                gpu_type_dist[gpu_index][prn_type] += 1       
                if(gpu_type_dist[gpu_index][prn_type] == 1):
                    fitness += 1                        
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
    """
    Combines two parent solutions into a new child solution using uniform crossover.
    """
    # Initialize child chromosome and GPUs 
    child_solution = np.full(total_prns, -1, dtype=int)
    child_gpu_vram = np.full(total_gpus, total_vram, dtype=int)
    child_gpu_type_dist = np.zeros((total_gpus, total_types), dtype=int)
    child_fitness = 0

    for prn_index in range(total_prns):
        prn_vram = prns[prn_index]['prn_vram']
        prn_type = prns[prn_index]['prn_type'] 

        # 50%-50% chance to receive each allocation from one of the parents
        if (rand.random() < 0.5):
            selected_gpu = parent1[prn_index]
        else:
            selected_gpu = parent2[prn_index]

        if(child_gpu_vram[selected_gpu] >= prn_vram):
            # If the chosen allocation is valid, transfer it to the child solution/chromosome
            child_solution[prn_index] = selected_gpu
            child_gpu_vram[selected_gpu] -= prn_vram
            child_gpu_type_dist[selected_gpu][prn_type] += 1
            if child_gpu_type_dist[selected_gpu][prn_type] == 1:
                child_fitness += 1
        else:
            # If the chosen allocation isnt valid, sort a valid GPU to allocate the PRN to
            valid_gpus = [gpu_idx for gpu_idx in range(total_gpus) if child_gpu_vram[gpu_idx] >= prn_vram]
            if valid_gpus:
                selected_gpu = rand.choice(valid_gpus)
                child_solution[prn_index] = selected_gpu
                child_gpu_vram[selected_gpu] -= prn_vram
                child_gpu_type_dist[selected_gpu][prn_type] += 1
                if child_gpu_type_dist[selected_gpu][prn_type] == 1:
                    child_fitness += 1
            else:
                # If couldn't find a valid recombination return a feasible solution
                return generate_feasible_solution(prns, total_prns, total_types, total_gpus, total_vram)
            
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
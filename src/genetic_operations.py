import random as rand
import numpy as np

def generate_valid_solution(prns, total_gpus, total_vram, total_types, total_prns):
    """
    Finds a valid solution for the given instance of the OGD problem.
    """
    sorted_indices = np.argsort(-prns['prn_vram'])                           # Get indices that would sort the PRNs by VRAM consumption (descending order)
    gpus_available_vram = np.full(total_gpus, total_vram, dtype=int)         # Track remaining VRAM for each GPU
    gpus_type_distribution = np.zeros((total_gpus, total_types), dtype=int)  # Track type distribution for each GPU (2D array: GPUs x Types)
    chromosome = np.full(total_prns, -1, dtype=int)                          # Solution array (-1 means unallocated) 
    chromosome_fitness = 0

    # First Fit Descending (FFD) greedy heuristic to try to find a valid solution
    for sorted_index in sorted_indices:
        prn_vram = prns[sorted_index]['prn_vram']
        prn_type = prns[sorted_index]['prn_type'] - 1

        # Check each GPU for sufficient VRAM        
        for gpu_index in range(total_gpus):            
            if (prn_vram <= gpus_available_vram[gpu_index]): 
                chromosome[sorted_index] = gpu_index                   
                gpus_available_vram[gpu_index] -= prn_vram             
                gpus_type_distribution[gpu_index][prn_type] += 1       
                if(gpus_type_distribution[gpu_index][prn_type] == 1):
                    chromosome_fitness -= 1                            
                break
        else:
            # If heuristic fails to find a valid solution run for the hills!
            return False

    return chromosome, gpus_available_vram, gpus_type_distribution, chromosome_fitness


def generate_initial_population(population_size, total_gpus, total_vram, total_types, total_prns, prns):  
    population = np.empty((population_size, total_prns), dtype=int)              # Array to store valid generated chromosomes
    vram_per_solution = np.empty((population_size, total_gpus), dtype=int)  # Array to store GPUs remaining VRAM for each chromosome
    type_distribution_per_solution = np.empty(((total_gpus, total_types), population_size), dtype=int)


    valid_solution, gpus = generate_valid_solution(prns, total_gpus, total_vram, total_types, total_prns)

    population[0] = valid_solution
    gpu_population[0] = gpus
    for i in range(1,population_size):
        population[i], gpu_population[i] = mutate_solution(valid_solution, int(0.5 * total_prns), prns, total_prns, gpus, total_gpus)

    return population, gpu_population


def mutate_solution(chromosome, total_gene_mutations, prns, total_prns, gpus, total_gpus):
    """
    Randomly mutates the input chromosome (solution) by changing the allocation of PRNs
    """
    for _ in range(total_gene_mutations):
        # Sort a random PRN to be reallocated
        prn_index = rand.randint(0, total_prns-1)
        prn_vram = prns[prn_index]['prn_vram']
        old_gpu_index = chromosome[prn_index]

        # Sort a valid GPU for the PRN to be reallocated to
        valid_gpus = [i for i in range(total_gpus) if gpus[i] >= prn_vram and i != old_gpu_index]
        if valid_gpus:  
            new_gpu_index = rand.choice(valid_gpus)

            gpus[old_gpu_index] += prn_vram  # Restore VRAM to the old GPU
            gpus[new_gpu_index] -= prn_vram  # Deduct VRAM from the new GPU

            chromosome[prn_index] = new_gpu_index

    return chromosome, gpus
    

def recombine_solutions(parent1, parent2, prns, total_prns, gpus1, gpus2, total_gpus, total_vram):
    """
    Combines two parent solutions into a new child solution using uniform crossover.
    """
    # Initialize child chromosome and GPUs 
    child_chromosome = np.full(total_prns, -1, dtype=int)
    child_gpus = np.full(total_gpus, total_vram, dtype=int)

    for prn_index in range(total_prns):
        prn_vram = prns[prn_index]['prn_vram']

        # 50%-50% chance to receive each allocation from one of the parents
        if (rand.random() < 0.5):
            selected_gpu = parent1[prn_index]
        else:
            selected_gpu = parent2[prn_index]

        if(child_gpus[selected_gpu] >= prn_vram):
            # If the chosen allocation is valid, transfer it to the child chromosome
            child_chromosome[prn_index] = selected_gpu
            child_gpus[selected_gpu] -= prn_vram
        else:
            # If the chosen allocation isnt valid, sort a valid GPU to allocate the PRN to
            valid_gpus = [gpu_idx for gpu_idx in range(total_gpus) if child_gpus[gpu_idx] >= prn_vram]
            if valid_gpus:
                selected_gpu = rand.choice(valid_gpus)
                child_chromosome[prn_index] = selected_gpu
                child_gpus[selected_gpu] -= prn_vram
            else:
                # If couldn't find a valid recombination return the first parent
                return parent1, gpus1
            
    return child_chromosome, child_gpus


def calculate_fitness(chromosome, prns, total_prns, total_gpus, total_types):
    """
    Calculate the quality of a given solution based on the distribution of PRN types along the GPUs
    """
    gpu_type_distribution = np.bool((total_gpus, total_types), dtype=bool)

    type_distribution = 0
    for prn_index in range(total_prns):
        prn_type = prns[prn_index]['prn_type'] - 1
        prn_gpu = chromosome[prn_index]

        if (not gpu_type_distribution[prn_gpu][prn_type]):
            gpu_type_distribution[prn_gpu][prn_type] = 1
            type_distribution += 1

    return type_distribution
        


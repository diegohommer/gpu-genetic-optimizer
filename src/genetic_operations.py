import random as rand
import numpy as np

def generate_valid_solution(prns, total_gpus, total_vram, total_types, total_prns):
    """
    Tries to find a valid solution for the given instance of the OGD problem
    """
    sorted_indices = np.argsort(-prns['prn_vram'])                           # Get indices that would sort the PRNs by VRAM consumption (descending order)
    gpus = np.full(total_gpus, total_vram, dtype=int)                        # Track remaining VRAM for each GPU
    gpus_type_distribution = np.zeros((total_gpus, total_types), dtype=int)  # Track type distribution for each GPU (2D array: GPUs x Types)
    chromosome = np.full(total_prns, -1, dtype=int)                          # Solution array (-1 means unallocated) 

    # First Fit Descending (FFD) heuristic to try to find a valid solution
    for sorted_index in sorted_indices:
        prn_vram = prns[sorted_index]['prn_vram']
        prn_type = prns[sorted_index]['prn_type'] - 1

        for gpu_index in range(total_gpus):
            gpu_remaining_vram = gpus[gpu_index]
            if (prn_vram <= gpu_remaining_vram): 
                chromosome[sorted_index] = gpu_index
                gpus[gpu_index] -= prn_vram
                gpus_type_distribution[gpu_index][prn_type] += 1
                break
        else:
            # If heuristic fails, run solver with a constant objetive function
            return False

    return chromosome, gpus


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
    # Initialize child chromosome and GPUs based on the parents
    child_chromosome = np.full(total_prns, -1, dtype=int)
    child_gpus = np.full(total_gpus, total_vram, dtype=int)

    for prn_index in range(total_prns):
        prn_vram = prns[prn_index]['prn_vram']

        # 50%-50% chance to receive the allocation from one of the parents
        if (rand.random() < 0.5):
            selected_gpu = parent1[prn_index]
        else:
            selected_gpu = parent2[prn_index]

        if(child_gpus[selected_gpu] >= prn_vram):
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
                return parent1, gpus1
            
    return child_chromosome, child_gpus


def calculate_fitness(chromosome, prns, total_prns, total_gpus, total_types):
    """
    Calculate the quality of a given solution based on the distribution of PRN types along the GPUs
    """
    gpu_type_distribution = np.zeros((total_gpus, total_types), dtype=bool)

    type_distribution = 0
    for prn_index in range(total_prns):
        prn_type = prns[prn_index]['prn_type'] - 1
        prn_gpu = chromosome[prn_index]

        if (not gpu_type_distribution[prn_gpu][prn_type]):
            gpu_type_distribution[prn_gpu][prn_type] = 1
            type_distribution += 1

    return type_distribution
        


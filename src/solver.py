import numpy as np

def run_genetic_algorithm(gpu_total, vram_total, types_total, prn_total, prns, params):
    initial_population = generate_initial_population(params.population_size, gpu_total, vram_total, prn_total, prns)
    


def generate_initial_population(population_size, gpu_total, vram_total, prn_total, prns):
    population = np.empty((population_size, prn_total), dtype=int)  
    valid_solution = generate_valid_solution(prns, gpu_total, vram_total, prn_total)

    population[0] = valid_solution
    
    return

def generate_valid_solution(prns, gpu_total, vram_total, prn_total):
    sorted_prns = prns[np.argsort(-prns['prn_vram'])]   # Sort PRNs by VRAM consumption (descending order)
    gpus = np.full(gpu_total, vram_total, dtype=int)    # Track remaining VRAM for each GPU
    cromossome = np.full(prn_total, -1, dtype=int)      # Solution array (-1 means unallocated) 

    # First Fit Descending (FFD) heuristic to try to find a valid solution
    for prn_index,prn in enumerate(sorted_prns):
        prn_vram = prn['prn_vram']

        for gpu_index in range(gpu_total):
            gpu_remaining_vram = gpus[gpu_index]
            if (prn_vram <= gpu_remaining_vram): 
                cromossome[prn_index] = gpu_index
                gpus[gpu_index] -= prn_vram
                break
        else:
            # If heuristic fails, run solver with a constant objetive function
            return False

    return cromossome

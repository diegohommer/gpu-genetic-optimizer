from genetic_operations import *

def generate_initial_population(population_size, total_gpus, total_vram, total_types, total_prns, prns):  
    population = np.empty((population_size, total_prns), dtype=int)      # List to store valid generated chromosomes
    gpu_population = np.empty((population_size, total_gpus), dtype=int)  # List to store associated GPUs states for each chromosome

    valid_solution, gpus = generate_valid_solution(prns, total_gpus, total_vram, total_types, total_prns)

    population[0] = valid_solution
    gpu_population[0] = gpus
    for i in range(1,population_size):
        population[i], gpu_population[i] = mutate_solution(valid_solution, int(0.5 * total_prns), prns, total_prns, gpus, total_gpus)

    return population, gpu_population

def run_genetic_algorithm(total_gpus, total_vram, total_types, total_prns, prns, params):
    population_size = params.population_size
    population, gpu_population = generate_initial_population(population_size, total_gpus, total_vram, total_types, total_prns, prns)

    i = 0
    while(i < 2):
        new_population = np.empty((population_size, total_prns), dtype=int)
        new_gpu_population = np.empty((population_size, total_gpus), dtype=int)
        for j in range(population_size):
            new_population[j], new_gpu_population[j] = recombine_solutions(population[0], population[1], prns, total_prns, gpu_population[0], gpu_population[1], total_gpus, total_vram)
        population = np.copy(new_population)
        i += 1

    print(population)
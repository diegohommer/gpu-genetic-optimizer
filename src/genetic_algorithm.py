"""
This module contains the implementation of the genetic algorithm for solving the OGD problem.

The `run_genetic_algorithm` function evolves a population of potential solutions over
multiple generations using selection, crossover, mutation, and elitism. The process continues
until either the maximum number of iterations is reached or stagnation is detected.

Function:
    run_genetic_algorithm: Executes the genetic algorithm to find the best solution.

Parameters:
    total_gpus (int): Total number of GPUs.
    total_vram (int): Total amount of VRAM available.
    total_types (int): Total number of GPU types.
    total_prns (int): Total number of PRNs.
    prns (list): List containing the PRNs.
    params (object): Configuration object containing genetic algorithm parameters such as
                     population size, mutation rate, elitism rate, etc.

Returns:
    None: The function prints the best solution, VRAM allocation, GPU type distribution,
          fitness score, and stagnation status at the end of the algorithm.
"""

import random as rand
import numpy as np
import time

from genetic_operations import \
    generate_initial_population, \
    mutate_solution, \
    crossover_solutions, \
    select_parents, \
    print_ga_outputs

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
    time_limit = params.time_limit
    max_iterations = params.max_iterations
    stagnation_limit = params.stagnation_limit
    seed = params.seed
    total_elites = int(population_size * elitism_rate)

    # Set random seed for reproducibility
    if seed:
        np.random.seed(seed)

    # Create initial population
    (
        population,
        gpu_vram_population,
        gpu_type_dist_population,
        fitness_population,
        best_solution
    ) = generate_initial_population(
        population_size, total_gpus, total_vram, total_types, total_prns, prns
    )

    start_time = time.time()
    i = 0
    stagnation_counter = 0
    while True:
        # Check for stop conditions
        if time_limit and (time.time() - start_time > time_limit):
            print("\nTime limit reached, stopping the algorithm.")
            break
        if max_iterations is not None and i >= max_iterations:
            print("\nMaximum iterations reached, stopping the algorithm.")
            break
        if stagnation_limit is not None and stagnation_counter >= stagnation_limit:
            print("\nStagnation limit reached, stopping the algorithm.")
            break

        # Valid generated solutions/chromosomes
        new_population = np.empty(
            (population_size, total_prns), dtype=int)

        # GPUs remaining VRAM per solution/chromosome
        new_gpu_vram_population = np.empty(
            (population_size, total_gpus), dtype=int)

        # GPUs type distribution per solution/chromosome
        new_gpu_type_dist_population = np.empty(
            (population_size, total_gpus, total_types), dtype=int)

        # Fitness per solution/chromosome
        new_fitness_population = np.empty(
            population_size, dtype=int)

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
            parent1_index, parent2_index = select_parents(
                fitness_population, selection_pressure, population_size
            )
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

    print_ga_outputs(
        population,
        gpu_vram_population,
        gpu_type_dist_population,
        fitness_population,
        best_solution,
        stagnated
    )
       
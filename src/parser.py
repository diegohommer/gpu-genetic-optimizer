import numpy as np
import argparse

def cmd_parser():
    """
    Parse command-line arguments for the Optimal GPU Distribution problem.
    
    Args:
        None: This function directly parses the command-line arguments.
    
    Returns:
        Namespace: An object containing the following parsed command-line arguments:
            output-file (str): File to save the best solution.
            population-size (int): Number of initial solutions in the population (default: 100).
            crossover-rate (float): Probability of recombining solutions during crossover (default: 0.5).
            mutation-rate (float): Probability of mutating solutions during evolution (default: 0.05).
            elitism-rate (float): Percentage of the best solutions retained in each generation (default: 0.1).
            selection-pressure (float): Constant selection pressure for parent selection. (default: 1.2)
            max-iterations (int): Maximum number of generations to evolve (default: 1000).
            stagnation-limit (int): Number of generations with no improvement before stopping early (default: 50).
            seed (int): Seed for the random number generator to ensure reproducibility (default: None).
    """
    parser = argparse.ArgumentParser(
        description="Solve an instance of the Optimal GPU Distribution problem with a genetic algorithm."
    )
    parser.add_argument("output_file", help="File to save the best solution.")
    parser.add_argument("-p", "--population-size", type=int, default=100, help="Number of initial solutions (default: 100).")
    parser.add_argument("-c", "--crossover-rate", type=float, default=0.8, help="Rate at which solutions are recombined (default: 0.5).")
    parser.add_argument("-m", "--mutation-rate", type=float, default=0.3, help="Rate at which solutions are mutated (default: 0.05).")
    parser.add_argument("-e", "--elitism-rate", type=float, default=0.1, help="Percentage of the best solutions retained (default: 0.1).")
    parser.add_argument("-P", "--selection-pressure", type=float, default=1.2, help="Constant selection pressure for parent selection. (default: 1.5)")
    parser.add_argument("-t", "--time-limit", type=int, default=1800, help="Maximum runtime in seconds. (default: None)")
    parser.add_argument("-i", "--max-iterations", type=int, default=None, help="Maximum number of generations (default: None).")
    parser.add_argument("-s", "--stagnation-limit", type=int, default=1000, help="Stop if no improvement after these many generations (default: None).")
    parser.add_argument("-S", "--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()

def instance_parser():
    """
    Parse an instance of the problem from user input.
    
    Args:
        None (uses input() to collect instance data).
        
    Returns:
        Tuple: A tuple containing the following:
            gpu_total: Total number of GPUs (int)
            vram_total: Total VRAM capacity of each GPU (int)
            types_total: Number of different types of PRNs (int)
            prns: Structured numpy array with PRN data (dtype: [('prn_type', 'i4'), ('prn_vram', 'i4')])
    """
    gpu_total = int(input())
    vram_total = int(input())
    types_total = int(input())
    prn_total = int(input())

    dtype = np.dtype([('prn_type', 'i4'), ('prn_vram', 'i4')])
    prns = np.zeros(prn_total, dtype=dtype)
    for i in range(prn_total):
        new_prn = input()
        prn_type, prn_vram = map(int, new_prn.split())
        prns[i] = (prn_type, prn_vram)

    return gpu_total, vram_total, types_total, prn_total, prns
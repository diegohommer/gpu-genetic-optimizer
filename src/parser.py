import numpy as np
import argparse

def cmd_parser():
    """
    Parse command-line arguments for the Optimal GPU Distribution problem.
    
    Args:
        None: This function directly parses the command-line arguments.
    
    Returns:
        Namespace: An object containing the following parsed command-line arguments:
            output_file: File to save the best solution (str).
            population_size: Number of initial solutions (int, default 100).
    """
    parser = argparse.ArgumentParser(
        description="Solve an instance of the Optimal GPU Distribution problem with a genetic algorithm."
    )
    parser.add_argument("output_file", help="File to save the best solution.")
    parser.add_argument("--population_size", type=int, required=False, default=100, help="Number of initial solutions (default 100).")
    parser.add_argument("--recombination_rate", type=float, required=False, default=0.5, help="Rate at which solutions are recombinated to generate new solutions.")
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
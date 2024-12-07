import argparse
import numpy as np

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
    return parser.parse_args()

def instance_parser():
    """
    Parse an instance of the problem from user input.
    
    Args:
        None (uses input() to collect data).
        
    Returns:
        Tuple: A tuple containing the following:
            gpu_total: Total number of GPUs (int)
            vram_total: Total VRAM capacity of each GPU (int)
            types_total: Number of different types of PRNs (int)
            prns: Structured numpy array with PRN data (dtype: [('prn_type', 'i4'), ('prn_vram', 'i4')])
    """
    gpu_total = int(input("Insert number of GPUs: "))
    vram_total = int(input("Insert VRAM capacity of GPUs: "))
    types_total = int(input("Insert number of different types: "))
    prn_total = int(input("Insert number of PRNs: "))

    dtype = np.dtype([('prn_type', 'i4'), ('prn_vram', 'i4')])
    prns = np.zeros(prn_total, dtype=dtype)
    for i in range(prn_total):
        new_prn = input(f"Insert {i + 1}º PRN (format: type vram): ")
        prn_type, prn_vram = map(int, new_prn.split())
        prns[i] = (prn_type, prn_vram)

    return gpu_total, vram_total, types_total, prns
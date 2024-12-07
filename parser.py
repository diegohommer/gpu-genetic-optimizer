import argparse
import numpy as np

def cmd_parser():
    """
    Parse command-line arguments for the Optimal GPU Distribution problem.
    """
    parser = argparse.ArgumentParser(
        description="Solve an instance of the Optimal GPU Distribution problem with a genetic algorithm."
    )
    parser.add_argument("output_file", help="File to save the best solution.")
    parser.add_argument("--population_size", type=int, required=False, default=100, help="Number of initial solutions (default 100).")
    parser.add_argument("--param2", type=int, required=False, help="Second parameter for the method.")
    return parser

def instance_parser():
    """
    Parse an instance of the problem from user input.
    Returns:
        gpu_total: Total number of GPUs
        vram_total: Total VRAM capacity of each GPU
        types_total: Number of different types of PRNs
        prns: Structured numpy array with PRN data
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
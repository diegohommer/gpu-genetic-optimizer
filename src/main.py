from parser import *
from genetic_algorithm import *

def main():
    # Receive algorithm parameters and instance
    params = cmd_parser()
    gpu_total, vram_total, types_total, prn_total, prns = instance_parser()

    #  Run genetic algorithm using received parameters and instace
    run_genetic_algorithm(gpu_total, vram_total, types_total, prn_total, prns, params)

if __name__ == "__main__":
    main()
from src.parser import *
from src.solver import *

def main():
    params = cmd_parser()
    print(params)

    gpu_total,vram_total,types_total,prns = instance_parser()
    solve_instance(gpu_total, vram_total, types_total, prns, params)

if __name__ == "__main__":
    main()
import argparse

def cmd_parser():
    # Set up command line parser
    parser = argparse.ArgumentParser(description="Solve an instance of the Optimal GPU Distribution problem with a genetic algorithm.")
    parser.add_argument("output_file", help="File to save the best solution.")
    parser.add_argument("--param1", type=int, required=False, help="First parameter for the method.")
    parser.add_argument("--param2", type=int, required=False, help="Second parameter for the method.")
    return parser
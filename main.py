from parser import *
from solver import *

def main():
    parser = cmd_parser()
    args = parser.parse_args()
    print(args)

if __name__ == "__main__":
    main()
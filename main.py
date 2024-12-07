from parser import *
from solver import *

def main():
    args = cmd_parser()
    print(args)

    print(instance_parser())

if __name__ == "__main__":
    main()
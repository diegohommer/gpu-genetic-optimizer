#!/bin/bash
python3 ./src/main.py \
    --population_size 1000 \
    --recombination_rate 0.4 \
    test.txt < example_instances/dog_5.txt
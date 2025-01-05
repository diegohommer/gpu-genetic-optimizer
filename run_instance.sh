python3 ./src/main.py \
    output.txt \
    --population-size 100 \
    --crossover-rate 0.8 \
    --mutation-rate 0.3 \
    --elitism-rate 0.1 \
    --selection-pressure 1.4 \
    --time-limit 1800 \
    --stagnation-limit 100 \
    < example_instances/dog_5.txt 
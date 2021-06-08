#!/bin/bash

# Script to reproduce results
# Running by typing the following cmd in terminal:
# chmod +x ./test.sh
# ./run_experiments.sh

for seed in 0 100 1000; do
    python ppo_runner.py \
    --env "LunarLander-v2" \
    --seed $seed \
    --epochs 150

    python ppo_runner.py \
    --env 'LunarLanderContinuous-v2' \
    --seed $seed \
    --epochs 150
done

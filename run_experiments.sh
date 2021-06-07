#!/bin/bash

# Script to reproduce results
# Running by typing the following cmd in terminal:
# chmod +x ./test.sh
# ./run_experiments.sh

for (( i = 0; i < 2; i++ )); do
    python ppo_runner.py \
    --env "HumanoidStandup-v2" \
    --seed $i \
    --epochs 500
done

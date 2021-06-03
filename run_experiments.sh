#!/bin/bash

# Script to reproduce results
# Running by typing the following cmd in terminal:
# chmod +x ./test.sh
# ./run_experiments.sh

for ((i=0;i<2;i+=1))
do
#	python main.py \
#	--policy "TD3" \
#	--env "HalfCheetah-v3" \
#	--seed $i
#
#	python main.py \
#	--policy "TD3" \
#	--env "Hopper-v3" \
#	--seed $i
#
#	python main.py \
#	--policy "TD3" \
#	--env "Walker2d-v3" \
#	--seed $i

	python ppo_runner.py \
	--env "Ant-v3" \
	--seed $i \
	--epochs 500

#	python main.py \
#	--policy "TD3" \
#	--env "Humanoid-v3" \
#	--seed $i
#
#	python main.py \
#	--policy "TD3" \
#	--env "InvertedPendulum-v2" \
#	--seed $i \
#	--start_timesteps 1000
#
#	python main.py \
#	--policy "TD3" \
#	--env "InvertedDoublePendulum-v2" \
#	--seed $i \
#	--start_timesteps 1000
#
#	python main.py \
#	--policy "TD3" \
#	--env "Reacher-v2" \
#	--seed $i \
#	--start_timesteps 1000
done
#!/bin/bash

python scripts/rsl_rl/train_race.py \
  --task Isaac-Quadcopter-Race-v0 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --headless \
  --logger wandb

echo "Training done. Shutting down..."
sudo shutdown -h now
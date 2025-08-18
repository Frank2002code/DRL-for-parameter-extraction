#!/bin/bash

source .venv/bin/activate
python train_ppo_tune.py --simulate_target_data --reduce_obs_err_dim --reward_norm --n_iterations 450 || echo "Training script failed" \
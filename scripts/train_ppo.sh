#!/bin/bash

source .venv/bin/activate
python train_ppo_tune.py --simulate_target_data --reduce_obs_err_dim --n_iterations 100 || echo "Training script failed" \
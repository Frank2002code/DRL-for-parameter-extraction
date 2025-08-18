#!/bin/bash

source .venv/bin/activate
python train_sac_tune.py --simulate_target_data --reduce_obs_err_dim --reward_norm --n_iterations 200 || echo "Training script failed"
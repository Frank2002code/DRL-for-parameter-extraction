#!/bin/bash

source .venv/bin/activate
python train_ppo_tune.py --reward_norm --reduce_obs_err_dim --n_iterations 200 || echo "Training script failed" \
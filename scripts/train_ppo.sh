#!/bin/bash

source .venv/bin/activate
python train_ppo_tune.py --reward_norm --n_iterations 100 || echo "Training script failed" \
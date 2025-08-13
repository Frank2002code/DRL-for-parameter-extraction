#!/bin/bash

source .venv/bin/activate
python train_sac.py --simulate_target_data --reward_norm || echo "Training script failed"
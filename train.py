from env.eehemt_env import EEHEMTEnv
from ray.rllib.algorithms.ppo import PPOConfig
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    csv_file_path = "/home/u5977862/DRL-on-parameter-extraction/data/S25E02A025WS_25C_GMVG.csv"
    tunable_parameters = {
        # === Threshold Voltage Related ===
        'Vto': {
            'min': -0.5,
            'max': 1.0,
            'delta': 0.01
        },
        'Gamma': {
            'min': 0.0,
            'max': 0.5,
            'delta': 0.001
        },
        'Vch': {
            'min': 0.5,
            'max': 3.0,
            'delta': 0.05
        },

        # === Transconductance & Current Gain ===
        'Gmmax': {
            'min': 0.05,
            'max': 0.5,
            'delta': 0.005
        },
        'Deltgm': {
            'min': 0.0,
            'max': 1.0,
            'delta': 0.01
        },

        # === Saturation Effects ===
        'Vsat': {
            'min': 0.1,
            'max': 2.0,
            'delta': 0.02
        },
        'Kapa': {
            'min': 0.0,
            'max': 0.5,
            'delta': 0.005
        },
        'Alpha': {
            'min': 0.001,
            'max': 0.2,
            'delta': 0.001
        },
        'Peff': {
            'min': 0.1,
            'max': 5.0,
            'delta': 0.1
        },

        # === Parasitic Resistances ===
        'Rs': {
            'min': 0.0,
            'max': 10.0,
            'delta': 0.2
        },
        'Rd': {
            'min': 0.0,
            'max': 10.0,
            'delta': 0.2
        },
    }
    va_file_path = "/home/u5977862/DRL-on-parameter-extraction/eehemt/eehemt114_2.va"

    parser.add_argument("--num_learners", type=int, default=2)
    parser.add_argument("--num_gpus_per_learner", type=float, default=0.5)
    
    args = parser.parse_args()

    # Configure.
    config = (
        PPOConfig()
        .environment(
            EEHEMTEnv,
            env_config={
                "csv_file_path": csv_file_path,
                "tunable_params_config": tunable_parameters,
                "va_file_path": va_file_path,
            },
        )
        .training(
            # num_learners=args.num_learners,
            # num_gpus_per_learner=args.num_gpus_per_learner,
            train_batch_size_per_learner=2000,
            lr=0.0004,
        )
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner
        )
    )

    # Build the Algorithm.
    algo = config.build_algo()

    # print(algo.train())
    algo.train()
    
    checkpoint_save_path = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_dir = algo.save_to_path()
    print(f"saved algo to {checkpoint_dir}")

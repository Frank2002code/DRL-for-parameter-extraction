# from ray import tune
import argparse

from ray.rllib.algorithms.ppo import PPOConfig

from env.eehemt_env import EEHEMTEnv, tunable_params_config

# import os
# import pprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # tunable_params_config = {
    #     # === Threshold Voltage Related ===
    #     "Vto": {"min": -0.5, "max": 1.0, "delta": 0.01},
    #     "Gamma": {"min": 0.0, "max": 0.5, "delta": 0.001},
    #     "Vch": {"min": 0.5, "max": 3.0, "delta": 0.05},
    #     # === Transconductance & Current Gain ===
    #     "Gmmax": {"min": 0.05, "max": 0.5, "delta": 0.005},
    #     "Deltgm": {"min": 0.0, "max": 1.0, "delta": 0.01},
    #     # === Saturation Effects ===
    #     "Vsat": {"min": 0.1, "max": 2.0, "delta": 0.02},
    #     "Kapa": {"min": 0.0, "max": 0.5, "delta": 0.005},
    #     "Alpha": {"min": 0.001, "max": 0.2, "delta": 0.001},
    #     "Peff": {"min": 0.1, "max": 5.0, "delta": 0.1},
    #     # === Parasitic Resistances ===
    #     "Rs": {"min": 0.0, "max": 10.0, "delta": 0.2},
    #     "Rd": {"min": 0.0, "max": 10.0, "delta": 0.2},
    # }

    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="/home/u5977862/DRL-on-parameter-extraction/data/S25E02A025WS_25C_GMVG.csv",
    )
    parser.add_argument(
        "--va_file_path",
        type=str,
        default="/home/u5977862/DRL-on-parameter-extraction/eehemt/eehemt114_2.va",
    )
    parser.add_argument("--test_modified", type=bool, default=True) 
    parser.add_argument("--num_learners", type=int, default=4)
    parser.add_argument("--num_gpus_per_learner", type=float, default=1.0)
    parser.add_argument("--num-iterations", type=int, default=200)

    args = parser.parse_args()

    # Configure.
    config = (
        PPOConfig()
        .environment(
            EEHEMTEnv,
            env_config={
                "csv_file_path": args.csv_file_path,
                "tunable_params_config": tunable_params_config,
                "va_file_path": args.va_file_path,
                "test_modified": args.test_modified,
            },
        )
        .training(
            train_batch_size_per_learner=2000,
            lr=0.0004,
        )
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner,
        )
    )

    # Build the Algorithm.
    algo = config.build_algo()

    # algo.train()
    for i in range(args.num_iterations):
        results = algo.train()

        print(f"--- Iteration: {i + 1}/{args.num_iterations} ---")
        # pprint.pprint(results)
    print("\nTraining completed.")

    # checkpoint_save_path = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_dir = algo.save_to_path()
    print(f"saved algo to {checkpoint_dir}")

    algo.stop()

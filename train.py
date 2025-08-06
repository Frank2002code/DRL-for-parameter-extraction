import argparse
import pprint

import ray
import torch as th
from ray.rllib.algorithms.ppo import PPOConfig

from env.eehemt_env import EEHEMTEnv_Norm, tunable_params_config
from utils.evaluate import eval_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--train_batch_size_per_learner", type=int, default=128)
    # parser.add_argument("--num_learners", type=int, default=4)
    # parser.add_argument("--num_gpus_per_learner", type=float, default=1.0)
    if th.cuda.device_count() == 4:
        num_learners = 4
        num_gpus_per_learner = 1.0
    elif th.cuda.device_count == 2:
        num_learners = 2
        num_gpus_per_learner = 1.0

    parser.add_argument("--n_iterations", type=int, default=300)

    args = parser.parse_args()

    # Configure.
    config = (
        PPOConfig()
        .environment(
            EEHEMTEnv_Norm,
            env_config={
                "csv_file_path": args.csv_file_path,
                "tunable_params_config": tunable_params_config,
                "va_file_path": args.va_file_path,
                "test_modified": args.test_modified,
            },
        )
        .env_runners(
            observation_filter="MeanStdFilter",  # Z-score norm better than L2 norm.
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            lr=0.0004,
        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        .evaluation(
            # We only need one evaluation worker for plotting
            evaluation_num_env_runners=1,
            # We will call `evaluate()` manually, so no interval is needed.
            evaluation_interval=None,
            # Point to our custom function
            custom_evaluation_function=eval_func,
            # Ensure evaluation is deterministic
            evaluation_config={"explore": False},
        )
    )

    # Build the Algorithm.
    algo = config.build_algo()

    for i in range(args.n_iterations):
        results = algo.train()
        print(f"--- Iteration: {i + 1}/{args.n_iterations} ---")
        episode_reward_mean = results.get('episode_reward_mean', float('nan'))
        print(f"Episode Reward Mean: {episode_reward_mean:.4f}")

    print("\n--- Training completed. ---")

    # === Evaluation ===
    final_results = algo.evaluate()
    print("\n--- Custom evaluation results ---")
    pprint.pprint(final_results)

    checkpoint_dir = "/home/u5977862/DRL-on-parameter-extraction/result/ckpt"
    checkpoint_dir = algo.save_to_path(checkpoint_dir)
    print(f"\nFinal algorithm checkpoint saved to: {checkpoint_dir}")

    algo.stop()
    ray.shutdown()
    print("\n--- Script finished. ---")

    algo.stop()

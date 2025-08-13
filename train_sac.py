import argparse
import os

import torch as th
from ray.rllib.algorithms.sac import SACConfig

from env.eehemt_env import EEHEMTEnv_Norm_Lgs, tunable_params_config

# from utils.callbacks import CustomEvalCallbacks
from utils.plot import PlotCurve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # === Env arguments ===
    parser.add_argument(
        "--va_file_path",
        type=str,
        default=os.getenv("VA_FILE_PATH", ""),
    )
    parser.add_argument(
        "--change_param_names",
        type=str,
        default=os.getenv("CHANGE_PARAM_NAMES", "Kapa"),
    )
    parser.add_argument(
        "--simulate_target_data",
        action="store_true",
        help="Whether to simulate target data",
    )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default=os.getenv("CSV_FILE_PATH", ""),
    )
    parser.add_argument("--test_modified", action="store_true")
    parser.add_argument("--reward_norm", action="store_true")
    parser.add_argument("--use_stagnation", action="store_true")

    # === Env runner arguments ===
    parser.add_argument("--num_env_runners", type=int, default=2)

    # === Training arguments ===
    parser.add_argument("--train_batch_size_per_learner", type=int, default=int(os.getenv("TRAIN_BATCH_SIZE_PER_LEARNER", 4096)))
    parser.add_argument("--num_epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 5)))
    parser.add_argument("--minibatch_size", type=int, default=int(os.getenv("MINIBATCH_SIZE", 512)))
    # parser.add_argument("--lr", type=float, default=float(os.getenv("LR", 5e-6)))
    parser.add_argument("--actor_lr", type=float, default=float(os.getenv("ACTOR_LR", 3e-5)))
    parser.add_argument("--critic_lr", type=float, default=float(os.getenv("CRITIC_LR", 3e-4)))
    parser.add_argument(
        "--entropy_coeff", type=float, default=float(os.getenv("ENTROPY_COEFF", 5e-3))
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--n_iterations", type=int, default=int(os.getenv("N_ITERATIONS", 100))
    )  # 100 -> 50

    # === Learner arguments ===
    # parser.add_argument("--num_learners", type=int, default=4)
    # parser.add_argument("--num_gpus_per_learner", type=float, default=1.0)
    if th.cuda.device_count() == 4:
        num_learners = 4
        num_gpus_per_learner = 1.0
    elif th.cuda.device_count() == 2:
        num_learners = 2
        num_gpus_per_learner = 1.0

    # === Evaluation arguments ===
    # parser.add_argument("--log_y", action="store_true")

    args = parser.parse_args()

    # === Algo Configure ===
    config = (
        SACConfig()
        .environment(
            EEHEMTEnv_Norm_Lgs,
            env_config={
                "va_file_path": args.va_file_path,
                "tunable_params_config": tunable_params_config,
                "change_param_names": args.change_param_names,
                "simulate_target_data": args.simulate_target_data,
                "csv_file_path": args.csv_file_path,
                "test_modified": args.test_modified,
                "reward_norm": args.reward_norm,
                "use_stagnation": args.use_stagnation,
            },
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            observation_filter="MeanStdFilter",  # Z-score norm better than L2 norm.
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            # lr=args.lr * num_learners,
            ### New
            actor_lr = args.actor_lr * num_learners,  # type: ignore
            critic_lr = args.critic_lr * num_learners,

        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        # .callbacks(CustomEvalCallbacks)
        .callbacks(PlotCurve)
        .evaluation(
            # We only need one evaluation worker for plotting
            evaluation_interval=1,
            evaluation_num_env_runners=1,
            evaluation_duration=1,  # Only one episode for evaluation
            evaluation_duration_unit="episodes",
            # custom_evaluation_function=eval_func,
            evaluation_config={"explore": False},
        )
    )

    # Build Algo instance
    algo = config.build_algo()

    for i in range(args.n_iterations):
        results = algo.train()
        print(f"--- Iteration: {i + 1}/{args.n_iterations} ---")

    print("\n==== Training completed. ====")

    checkpoint_dir = "/home/u5977862/DRL-on-parameter-extraction/result/sac/ckpt"
    checkpoint_dir = algo.save_to_path(checkpoint_dir)
    print(f"\nFinal algorithm checkpoint saved to: {checkpoint_dir}")

    algo.stop()
    print("\n==== Script finished. ====")

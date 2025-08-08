import argparse
import os

import torch as th
from ray.rllib.algorithms.ppo import PPOConfig

from env.eehemt_env import EEHEMTEnv_Norm_Vtos, tunable_params_config

# from utils.callbacks import CustomEvalCallbacks
from utils.plot import PlotCurve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # === Env arguments ===
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="/home/u5977862/DRL-on-parameter-extraction/data/S25E02A025WS_25C_GMVG.csv",
    )
    parser.add_argument(
        "--va_file_path",
        type=str,
        default="/home/u5977862/DRL-on-parameter-extraction/env/eehemt/eehemt114_2.va",
    )
    parser.add_argument("--test_modified", type=bool, default=True)

    # === Env runner arguments ===
    parser.add_argument("--num_env_runners", type=int, default=2)

    # === Training arguments ===
    parser.add_argument("--train_batch_size_per_learner", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=512)
    parser.add_argument(
        "--entropy_coeff", type=float, default=float(os.getenv("ENTROPY_COEFF", 5e-3))
    )
    parser.add_argument(
        "--n_iterations", type=int, default=int(os.getenv("N_ITERATIONS", 100))
    )  # 100 -> 50

    # === Learner arguments ===
    # parser.add_argument("--num_learners", type=int, default=4)
    # parser.add_argument("--num_gpus_per_learner", type=float, default=1.0)
    if th.cuda.device_count() == 4:
        num_learners = 4
        num_gpus_per_learner = 1.0
    elif th.cuda.device_count == 2:
        num_learners = 2
        num_gpus_per_learner = 1.0

    args = parser.parse_args()

    # === Algo Configure ===
    config = (
        PPOConfig()
        .environment(
            EEHEMTEnv_Norm_Vtos,
            env_config={
                "csv_file_path": args.csv_file_path,
                "tunable_params_config": tunable_params_config,
                "va_file_path": args.va_file_path,
                "test_modified": args.test_modified,
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
            lr=0.00015 * num_learners,
            ### New
            entropy_coeff=args.entropy_coeff,  # type: ignore
        )
        .rl_module()
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

    print("\n--- Training completed. ---")

    # === Evaluation ===
    # print("\n--- Running final evaluation ---")
    # eval_results = algo.evaluate()
    # try:
    #     first_episode_metrics = eval_results["evaluation"]["custom_metrics_per_episode"][0]
    #     print("\n--- Keys available in custom_metrics_per_episode ---")
    #     print(first_episode_metrics.keys())
    #     print("--------------------------------------------------\n")

    #     final_rmspe = first_episode_metrics["final_rmspe"]
    #     plot_data = first_episode_metrics["plot_data"]

    #     print(f"Final RMSPE: {final_rmspe:.4f}")
    #     plot_iv_curve(
    #         plot_data=plot_data,
    #         plot_initial=True,
    #         plot_modified=True,
    #         plot_current=True,
    #         save_path="results/final_iv_curve.png"
    #     )
    # except (KeyError, IndexError) as e:
    #     print(f"\nCould not extract custom metrics for plotting. Error: {e}")
    #     pprint.pprint(eval_results)

    checkpoint_dir = "/home/u5977862/DRL-on-parameter-extraction/result/ckpt"
    checkpoint_dir = algo.save_to_path(checkpoint_dir)
    print(f"\nFinal algorithm checkpoint saved to: {checkpoint_dir}")

    print("\n--- Script finished. ---")

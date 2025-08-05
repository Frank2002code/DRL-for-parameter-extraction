# mypy: disable-error-code="union-attr"
import argparse
import os

import ray
import torch as th
from dotenv import load_dotenv
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_action_dist import TorchSquashedGaussian
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy

from env.eehemt_env import EEHEMTEnv, tunable_params_config

load_dotenv()
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "")


def restore_and_plot(
    config: AlgorithmConfig,
    output_filepath: str,
) -> None:
    ray.init(ignore_reinit_error=True)

    print(f"From checkpoint {CHECKPOINT_PATH}")

    # 1. Restore the algorithm from the checkpoint
    try:
        # algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
        algo = config.build_algo()
        algo.restore_from_path(CHECKPOINT_PATH)
        print("Algorithm restored.")
    except Exception as e:
        print(f"Restore failed. Please check the checkpoint path. Error: {e}")
        ray.shutdown()
        return

    # 2. Get the new env instance
    env_config = algo.get_config().env_config
    # print(env_config)
    env = algo.env_creator(env_config)
    print(f"Get env success: {type(env)}")

    # 3. Let agent run one episode
    module = algo.get_module()
    print("RLModule obtained.")

    # dist_class = algo.config.policies["default_policy"].action_distribution_class
    dist_class = TorchSquashedGaussian

    print("Running one episode with the trained agent...")
    observation, info = env.reset()
    terminated, truncated = False, False

    while not terminated and not truncated:
        #    [observation] -> [[observation]]
        obs_tensor = th.from_numpy(observation).unsqueeze(0)
        batch = {SampleBatch.OBS: obs_tensor}

        fwd_out = module.forward_inference(batch)
        action_dist = dist_class(fwd_out[SampleBatch.ACTION_DIST_INPUTS], module)
        action_tensor = action_dist.deterministic_sample()
        action = convert_to_numpy(action_tensor[0])

        observation, reward, terminated, truncated, info = env.step(action)

    print("Episode finished. The environment is now in its final state.")

    # 4. Plot the I-V curve
    print("Plotting I-V curve...")
    env.plot_iv_curve(
        plot_initial=True,
        plot_modified=False,
        plot_current=True,
        save_path=output_filepath,
    )

    ray.shutdown()
    print(f"\nMission complete! Result saved to ./{output_filepath}")


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
    args = parser.parse_args()
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
    )
    output_filepath = (
        "/home/u5977862/DRL-on-parameter-extraction/result/trained_iv_curve.png"
    )
    restore_and_plot(
        config,
        output_filepath,
    )

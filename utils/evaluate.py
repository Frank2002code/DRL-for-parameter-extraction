import os
import pprint

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.typing import ResultDict

from env.eehemt_env import tunable_params_config

def eval_func(
    algorithm: Algorithm, eval_workers: EnvRunnerGroup
) -> ResultDict:
    """
    Custom evaluation function that runs one episode, plots the I-V curve,
    and returns final metrics.
    """
    print("\n--- Running final evaluation and plotting I-V curve... ---")

    # 1. Get the local evaluation worker, its environment, and the trained policy.
    local_worker = eval_workers.local_env_runner
    env = local_worker.env
    policy = algorithm.get_policy()

    # 2. Run a single, deterministic episode to find the best parameters.
    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action, _, _ = policy.compute_single_action(observation=obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("Final evaluation episode finished.")
    print(f"Final RMSPE: {info['current_rmspe']:.4f}")
    print("Final Tunable Parameters:")
    final_tunable_params = {
        k: info["current_params"][k] for k in tunable_params_config.keys()
    }
    pprint.pprint(final_tunable_params)

    # 3. Plot the I-V curve using the environment's final state.
    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "final_iv_curve.png")

    env.plot_iv_curve(
        plot_initial=True, plot_modified=True, plot_current=True, save_path=save_path
    )

    # 4. Return a dictionary of final metrics.
    return {
        "final_episode_reward": total_reward,
        "final_rmspe": info["current_rmspe"],
    }

import os

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from dotenv import load_dotenv

load_dotenv()
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "")

def restore_and_plot() -> None:
    ray.init(ignore_reinit_error=True)

    print(f"From checkpoint {CHECKPOINT_PATH}")

    # 1. Restore the algorithm from the checkpoint
    try:
        algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
        print("Algorithm restored.")
    except Exception as e:
        print(f"Restore failed. Please check the checkpoint path. Error: {e}")
        ray.shutdown()
        return

    # 2. Get the environment from the algorithm
    env = algo.workers.local_worker().env
    print(f"Get env success: {type(env)}")

    # 3. Plot the I-V curve
    print("Plotting I-V curve...")
    output_filename = "trained_iv_curve.png"
    env.plot_iv_curve(
        plot_initial=True,
        plot_modified=False,
        plot_current=True,
        save_path=output_filename, 
    )


    ray.shutdown()
    print(f"\nMission complete! Result saved to ./{output_filename}")


if __name__ == "__main__":
    restore_and_plot()

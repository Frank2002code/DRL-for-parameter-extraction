import os

import matplotlib.pyplot as plt
from ray.rllib.algorithms.callbacks import DefaultCallbacks


def plot_iv_curve(
    plot_data: dict,
    plot_initial: bool = True,
    plot_modified: bool = True,
    plot_current: bool = True,
    save_path: str | None = None,
):
    """
    Use pre-calculated data to plot the I-V curve.
    """
    output_dir = os.path.dirname(save_path) if save_path else "results"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))

    # Plot measured data from the dictionary
    plt.plot(
        plot_data["vgs"], plot_data["i_meas"], "ko", label="Measured Data (Target)"
    )

    # Plot simulated curves based on options
    if plot_initial:
        plt.plot(
            plot_data["vgs"],
            plot_data["i_sim_initial"],
            "b--",
            label=f"Simulated (Initial Params, Vto={plot_data['vto']:.2f})",
        )

    if plot_modified:
        plt.plot(
            plot_data["vgs"],
            plot_data["i_sim_modified"],
            "g-.",
            label=f"Simulated (Modified Initial Params, Vto={plot_data['vto']:.2f})",
        )

    if plot_current:
        plt.plot(
            plot_data["vgs"],
            plot_data["i_sim_current"],
            "r-",
            label=f"Simulated (Current Params, Vto={plot_data['vto']:.2f})",
        )

    # Style the plot
    plt.title("I-V Curve Comparison")
    plt.xlabel("Gate Voltage (Vg) [V]")
    plt.ylabel("Log Drain Current (Id) [A]")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"I-V curve plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


### New
class PlotCurve(DefaultCallbacks):
    """
    RLlib Callback for plotting I-V curves at the end of each episode.
    It fetches static data (vgs, i_meas, etc.) only once and stores it.
    """
    def __init__(self):
        super().__init__()
        self.plot_data = None

    def on_environment_created(self, *, env_runner, metrics_logger = None, env, env_context, **kwargs):
        # print(f"Wrapper env type: {type(env).__name__}")
        actual_env = env.envs[0].unwrapped
        # print(f"Actual env type: {type(actual_env).__name__}")

        if hasattr(actual_env, "_get_plot_data"):
            # Fetch static plot data only once
            print("Fetching static plot data from the environment...")
            self.plot_data = actual_env._get_plot_data()
        else:
            print("Warning: Environment does not have '_get_plot_data' method.")
            return

    def on_episode_end(
        self,
        *,
        episode,
        **kwargs,
    ) -> None:
        last_info = episode.infos[-1]
        if "i_sim_current" in last_info:
            rmspe = last_info["final_rmspe"]
            i_sim_current = last_info["i_sim_current"]
            plot_data = self.plot_data.update(i_sim_current)

            # episode.custom_data["final_rmspe"] = rmspe
            # episode.custom_data["plot_data"] = plot_data
            print(f"\n=====Final RMSPE: {rmspe:.4f}=====")
            plot_iv_curve(
                plot_data=plot_data,
                save_path="result/final_iv_curve_2.png",
            )

import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from ray.rllib.algorithms.callbacks import RLlibCallback

load_dotenv()


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
def plot_all_vto_iv_curve(
    vto_values: list,
    plot_data: dict,
    plot_dir: str,
):
    """
    Plots and saves individual I-V curves for each Vto condition.

    Args:
        vto_values (list): A list containing all Vto float values.
        plot_data (dict): A dictionary containing static plotting data,
                          such as 'vgs', 'i_meas_dict', etc.
        plot_dir (str): The directory path to save the plots.
    """
    # Get static data from plot_data
    vgs = plot_data["vgs"]
    i_meas_dict = plot_data["i_meas_dict"]
    i_sim_initial_matrix = plot_data["i_sim_initial_matrix"]
    i_sim_modified_matrix = plot_data["i_sim_modified_matrix"]
    i_sim_current_matrix = plot_data["i_sim_current_matrix"]

    # Iterate through each Vto and its corresponding index
    for i, vto in enumerate(vto_values):
        plt.figure(figsize=(10, 7))

        # 1. Plot the target data (Measured)
        plt.plot(vgs, i_meas_dict[vto], "ko", label="Measured (Target)")

        # 2. Plot the simulated curve with initial parameters
        plt.plot(
            vgs,
            i_sim_initial_matrix[i, :],  # Get the i-th row
            "b--",
            label="Simulated (Initial)",
        )

        # 3. Plot the simulated curve with modified initial parameters
        plt.plot(
            vgs,
            i_sim_modified_matrix[i, :],  # Get the i-th row
            "g-.",
            label="Simulated (Modified)",
        )

        # 4. Plot the simulated curve with the agent's final parameters
        plt.plot(
            vgs,
            i_sim_current_matrix[i, :],  # Get the i-th row
            "r-",
            label="Simulated (Current)",
        )

        # Set the plot style
        plt.title(f"I-V Curve Comparison (Vto = {vto:.2f})")
        plt.xlabel("Gate Voltage (Vg) [V]")
        plt.ylabel("Log Drain Current (Id) [A]")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()

        # Generate filename based on Vto and save the plot
        save_path = os.path.join(plot_dir, f"final_iv_curve_vto_{vto:.2f}.png")
        plt.savefig(save_path)

        # Close the figure to release memory, which is crucial when plotting in a loop
        plt.close()


### New
class PlotCurve(RLlibCallback):
    """
    RLlib Callback for plotting I-V curves at the end of each episode.
    It fetches static data (vgs, i_meas, etc.) only once and stores it.
    """

    def __init__(self):
        super().__init__()
        self.plot_dir = os.getenv("PLOT_DIR", "result")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.plot_data = None
        ### New
        self.vto_values = []  # Store Vto values for plotting

    def on_environment_created(
        self, *, env_runner, metrics_logger=None, env, env_context, **kwargs
    ):
        # print(f"Wrapper env type: {type(env).__name__}")
        actual_env = env.envs[
            0
        ].unwrapped  # type(actual_env).__name__ = EEHEMTEnv_Norm_Vtos
        # print(f"Actual env type: {type(actual_env).__name__}")

        if self.plot_data is None:
            print("Fetching static plot data from the environment...")
            if hasattr(actual_env, "_get_plot_data"):
                # Fetch static plot data only once
                # self.plot_data = actual_env._get_plot_data()
                self.plot_data = actual_env._get_plot_data_matrix()
                self.vto_values = actual_env.vto_values
            else:
                print("Warning: Environment does not have '_get_plot_data' method.")
                self.plot_data = {}
                return

    def on_episode_end(
        self,
        *,
        episode,
        **kwargs,
    ) -> None:
        last_info = episode.infos[-1]
        if "i_sim_current_matrix" in last_info:
            rmspe = last_info["final_rmspe"]
            i_sim_current_matrix = last_info["i_sim_current_matrix"]
            self.plot_data.update(i_sim_current_matrix)  # Return None

            print(
                f"\n===== Final RMSPE: {rmspe:.4f} =====\nStarting to plot {len(self.vto_values)} curves..."
            )
            # plot_iv_curve(
            #     plot_data=self.plot_data,
            #     save_path=os.path.join(self.plot_dir, "final_iv_curve.png"),
            # )
            plot_all_vto_iv_curve(
                vto_values=self.vto_values,
                plot_data=self.plot_data,
                plot_dir=self.plot_dir,
            )
            print(f"===== All plots saved in '{self.plot_dir}' directory. =====")
import os

import matplotlib.pyplot as plt
import numpy as np
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
def plot_all_lg_iv_curve(
    lg_values: list,
    plot_data: dict,
    plot_dir: str,
):
    """
    Plots and saves individual I-V curves for each lg condition.

    Args:
        lg_values (list): A list containing all lg float values.
        plot_data (dict): A dictionary containing static plotting data,
                          such as 'vgs', 'i_meas_dict', etc.
        plot_dir (str): The directory path to save the plots.
    """
    # Get static data from plot_data
    vgs = plot_data["vgs"]
    i_meas_dict = plot_data["i_meas_dict"]
    i_sim_init_matrix = plot_data["i_sim_init_matrix"]
    # i_sim_modified_matrix = plot_data["i_sim_modified_matrix"]
    i_sim_current_matrix = plot_data["i_sim_current_matrix"]

    plt.figure(figsize=(12, 8))

    # Iterate through each lg and its corresponding index
    for i, lg in enumerate(lg_values):
        # 1. Plot the target data (Measured)
        plt.plot(vgs, i_meas_dict[lg], "o", label=f"Target (lg={lg})")

        # 2. Plot the simulated curve with initial parameters
        plt.plot(
            vgs,
            i_sim_init_matrix[i, :],  # Get the i-th row
            "b--",
            label=f"Initial (lg={lg})",
        )

        # 3. Plot the simulated curve with the agent's final parameters
        plt.plot(
            vgs,
            i_sim_current_matrix[i, :],  # Get the i-th row
            "r-",
            label=f"Current (lg={lg})",
        )

    # Set the plot style
    # plt.title(f"I-V Curve Comparison (lg = {lg:.2f})")

    plt.title("I-V Curve Comparison for All lg Values")
    plt.xlabel("Gate Voltage (Vg) [V]")
    plt.ylabel("Log Drain Current (Id) [A]")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # Save the plot
    save_path = os.path.join(plot_dir, "final_iv_curve_all_lg.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"==== I-V curve plot saved in {save_path} ====")


def plot_all_lg_iv_curve_colormap(
    lg_values: list,
    plot_data: dict,
    plot_dir: str,
    log_y: bool = True,
):
    """
    Plots and saves I-V curves for all lg conditions on a single graph.
    Each curve type (Target, Initial, Current) has its own color gradient.

    Args:
        lg_values (list): A list containing all lg float values.
        plot_data (dict): A dictionary containing static plotting data.
        plot_dir (str): The directory path to save the plots.
    """
    # --- Get static data from plot_data ---
    vgs = plot_data["vgs"]
    i_meas_dict = plot_data["i_meas_dict"]
    i_sim_init_matrix = plot_data["i_sim_init_matrix"]
    i_sim_current_matrix = plot_data["i_sim_current_matrix"]

    plt.figure(figsize=(12, 8))

    # --- Create distinct color maps for each curve type ---
    # We generate a list of colors for each type of curve.
    # Using np.linspace(0.3, 1, ...) ensures colors are not too light.
    num_curves = len(lg_values)
    target_colors = plt.get_cmap("Blues")(np.linspace(0.3, 1, num_curves))
    initial_colors = plt.get_cmap("Greens")(np.linspace(0.3, 1, num_curves))
    current_colors = plt.get_cmap("Reds")(np.linspace(0.3, 1, num_curves))

    # --- Iterate through each lg and plot with gradient colors ---
    for i, lg in enumerate(lg_values):
        label_target = "Target" if i == len(lg_values) - 1 else None
        label_initial = "Initial" if i == len(lg_values) - 1 else None
        label_current = "Current" if i == len(lg_values) - 1 else None
        # 1. Plot the target data (Measured) using the 'Blues' colormap.
        plt.plot(
            vgs,
            i_meas_dict[lg],
            marker="o",  # Set marker style
            linestyle="None",  # Do not connect the dots
            color=target_colors[i],
            label=label_target,  # Only label the first curve for legend
        )

        # 2. Plot the initial simulation using the 'Greens' colormap.
        plt.plot(
            vgs,
            i_sim_init_matrix[i, :],
            linestyle="--",  # Set line style to dashed
            color=initial_colors[i],
            label=label_initial,
        )

        # 3. Plot the current simulation using the 'Reds' colormap.
        plt.plot(
            vgs,
            i_sim_current_matrix[i, :],
            linestyle="-",  # Set line style to solid
            color=current_colors[i],
            label=label_current,
        )

    # --- Set the plot style and labels ---
    plt.title("I-V Curve Comparison for All lg Values")
    plt.xlabel("Gate Voltage (Vg) [V]")
    if log_y:
        plt.ylabel("Log Drain Current (Id) [A]")
        plt.yscale("log")
    else:
        plt.ylabel("Drain Current (Id) [A]")
    plt.grid(True, which="both", ls="--", alpha=0.7)

    # Place legend outside the plot area to avoid covering data
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend

    # --- Save the plot ---
    save_path = os.path.join(plot_dir, "final_iv_curve_all_lg.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"==== I-V curve plot saved in {save_path} ====")


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
        self.lg_values = []  # Store lg values for plotting

    def on_environment_created(
        self, *, env_runner, metrics_logger=None, env, env_context, **kwargs
    ):
        # print(f"Wrapper env type: {type(env).__name__}")
        actual_env = env.envs[
            0
        ].unwrapped  # type(actual_env).__name__ = EEHEMTEnv_Norm_Lgs
        # print(f"Actual env type: {type(actual_env).__name__}")

        if self.plot_data is None:
            print("Fetching static plot data from the environment...")
            if hasattr(actual_env, "_get_plot_data_matrix"):
                # Fetch static plot data only once
                # self.plot_data = actual_env._get_plot_data()
                self.plot_data = actual_env._get_plot_data_matrix()
                self.lg_values = actual_env.lg_values
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
                f"\n===== Final RMSPE: {rmspe:.4f} =====\nStarting to plot {3 * len(self.lg_values)} curves..."
            )
            # plot_iv_curve(
            #     plot_data=self.plot_data,
            #     save_path=os.path.join(self.plot_dir, "final_iv_curve.png"),
            # )
            plot_all_lg_iv_curve_colormap(
                lg_values=self.lg_values,
                plot_data=self.plot_data,
                plot_dir=self.plot_dir,
                log_y=False,
            )
            # print(f"===== All plots saved in '{self.plot_dir}' directory. =====")

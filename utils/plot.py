import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env.eehemt_env import tunable_params_names, CHANGE_PARAM_NAMES

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


### New
def plot_all_ugw_n_iv_curve_colormap(
    ugw_n_values: list,
    plot_data: dict,
    plot_dir: str,
    log_y: bool = True,
):
    """
    Plots and saves I-V curves for all (Ugw, NOF) conditions on a single graph.
    Each curve type (Target, Initial, Current) has its own color gradient.

    Args:
        ugw_n_values (list): A list containing all (Ugw, NOF) float values.
        plot_data (dict): A dictionary containing static plotting data.
        plot_dir (str): The directory path to save the plots.
    """
    # === Get static data from plot_data ===
    vgs = plot_data["vgs"]
    i_meas_dict = plot_data["i_meas_dict"]
    i_sim_init_matrix = plot_data["i_sim_init_matrix"]
    i_sim_current_matrix = plot_data["i_sim_current_matrix"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # === Create distinct color maps for each curve type ===
    # We generate a list of colors for each type of curve.
    # Using np.linspace(0.5, 1, ...) ensures colors are not too light.
    num_curves = len(ugw_n_values)
    target_colors = plt.get_cmap("Blues")(np.linspace(0.5, 1, num_curves))
    initial_colors = plt.get_cmap("Greens")(np.linspace(0.5, 1, num_curves))
    current_colors = plt.get_cmap("Reds")(np.linspace(0.5, 1, num_curves))

    # === Iterate through each (Ugw, NOF) pair and plot with gradient colors ===
    for i, ugw_n in enumerate(ugw_n_values):
        label_target = "Target" if i == len(ugw_n_values) - 1 else None
        label_initial = "Initial" if i == len(ugw_n_values) - 1 else None
        label_current = "Final" if i == len(ugw_n_values) - 1 else None
        # 1. Plot the target data (Measured) using the 'Blues' colormap.
        ax.plot(
            vgs,
            i_meas_dict[ugw_n],
            marker="o",
            linestyle="None",
            color=target_colors[i],
            label=label_target,
            ms=3,
        )

        # 2. Plot the initial simulation using the 'Greens' colormap.
        ax.plot(
            vgs,
            i_sim_init_matrix[i, :],
            linestyle="--",  # Set line style to dashed
            color=initial_colors[i],
            label=label_initial,
        )

        # 3. Plot the current simulation using the 'Reds' colormap.
        ax.plot(
            vgs,
            i_sim_current_matrix[i, :],
            linestyle="-",  # Set line style to solid
            color=current_colors[i],
            label=label_current,
        )

    # === Set the plot style and labels ===
    ax.set_title(f"I-V Curve Comparison for All {', '.join(CHANGE_PARAM_NAMES)} Values")
    ax.set_xlabel("Gate Voltage (Vg) [V]")
    if log_y:
        ax.set_ylabel("Log Drain Current (Id) [A]")
        ax.set_yscale("log")
        save_path = os.path.join(
            plot_dir, f"final_iv_curve_all_{'_'.join(CHANGE_PARAM_NAMES)}_log.png"
        )
    else:
        ax.set_ylabel("Drain Current (Id) [A]")
        save_path = os.path.join(
            plot_dir, f"final_iv_curve_all_{'_'.join(CHANGE_PARAM_NAMES)}.png"
        )
    plt.grid(True, which="both", ls="--", alpha=0.7)

    ax.grid(True, which="both", ls="--", alpha=0.7)
    ax.legend(loc="best")

    # === Save the plot ===
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"==== I-V curves plot saved in {save_path} ====")


### New
class PlotCurve(DefaultCallbacks):
    """
    RLlib Callback for plotting I-V curves at the end of each episode.
    It fetches static data (vgs, i_meas, etc.) only once and stores it.
    """

    def __init__(self):
        super().__init__()
        self.plot_dir = os.getenv("PLOT_DIR", "result/iv-curve")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir, exist_ok=True)
        self.plot_data = None
        ### New
        self.ugw_n_values = []  # Store lg values for plotting

    def on_environment_created(
        self, *, env_runner, metrics_logger=None, env, env_context, **kwargs
    ):
        actual_env = env.envs[0].unwrapped  # type(actual_env).__name__ = EEHEMTEnv_Norm_Lgs

        if self.plot_data is None:
            print("\nFetching static plot data from the environment...\n")
            if hasattr(actual_env, "_get_plot_data_matrix"):
                # Fetch static plot data only once
                self.plot_data = actual_env._get_plot_data_matrix()
                self.ugw_n_values = actual_env.ugw_n_values
            else:
                print("Warning: Environment does not have '_get_plot_data' method.")
                self.plot_data = {}
                return

    def on_evaluate_end(
        self, *, algorithm, metrics_logger=None, evaluation_metrics, **kwargs
    ):
        # algorithm: PPO(env=<class 'env.eehemt_env.EEHEMTEnv_Norm_Lgs'>; env-runners=10; learners=4; multi-agent=False)
        # metrics_logger: <ray.rllib.utils.metrics.metrics_logger.MetricsLogger object at 0x7f4080566350>
        # evaluation_metrics: {'env_runners': {'module_to_env_connector': {'timers': {'connectors': {'listify_data_for_vector_env': 3.677929740308673e-05,
        # 'remove_single_ts_time_rank_from_batch': 2.2118753837598408e-06, 'get_actions': 7.603191273219271e-05, 'tensor_to_numpy': 6.033979039501661e-05,
        # 'un_batch_to_individual_items': 2.0221763372970588e-05, 'normalize_and_clip_actions': 6.799664383943636e-05}}, 'connector_pipeline_timer': 0.0003552631200052729},
        # 'env_to_module_connector': {'timers': {'connectors': {'add_states_from_episodes_to_batch': 4.732462731121214e-06, 'batch_individual_items': 2.2336920543336885e-05,
        # 'add_observations_from_episodes_to_batch': 9.787067230587847e-06, 'add_time_dim_to_batch_and_zero_pad': 9.152046474831904e-06, 'numpy_to_tensor': 5.582577872646182e-05}},
        # 'connector_pipeline_timer': 0.00016570912673812504}, 'agent_episode_return_mean': {'default_agent': -3608.688799738884},
        # 'rlmodule_inference_timer': 0.00017641018404863687, 'num_agent_steps_sampled': {'default_agent': 1000},
        # 'module_episode_return_mean': {'default_policy': -3608.688799738884}, 'num_module_steps_sampled_lifetime': {'default_policy': 1000},
        # 'num_env_steps_sampled_lifetime': 1000, 'episode_return_min': -3608.688799738884, 'episode_len_mean': 1000.0, 'num_episodes_lifetime': 1,
        # 'num_episodes': 1, 'num_env_steps_sampled': 1000, 'episode_duration_sec_mean': 2.0138601511716843, 'episode_len_max': 1000,
        # 'episode_return_mean': -3608.688799738884, 'weights_seq_no': 1.0, 'num_agent_steps_sampled_lifetime': {'default_agent': 1000},
        # 'env_to_module_sum_episodes_length_in': 901.0042739534927, 'episode_return_max': -3608.688799738884, 'num_module_steps_sampled': {'default_policy': 1000},
        # 'episode_len_min': 1000, 'env_step_timer': 0.0011205974719583366, 'env_reset_timer': 0.001857757568359375,
        # 'env_to_module_sum_episodes_length_out': 901.0042739534927, 'sample': 3.3040749318897724}}
        pass

    ### New
    def on_episode_start(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        # episode.custom_data["Vto"] = []
        tunable_params = {name: [] for name in tunable_params_names}
        episode.custom_data.update(tunable_params)

    def on_episode_step(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        current_params = env.envs[0].unwrapped.current_params
        # print(f"current_params: {current_params}")
        # episode.custom_data["Vto"].append(current_params["Vto"])
        for param_name in tunable_params_names:
            episode.custom_data[param_name].append(current_params[param_name])

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        last_info = episode.infos[-1]
        if "i_sim_current_matrix" in last_info:
            current_rmspe = last_info["current_rmspe"]
            i_sim_current_matrix = last_info["i_sim_current_matrix"]
            self.plot_data.update(i_sim_current_matrix)  # type: ignore
            # Return None

            print(
                f"\nFinal RMSPE: {current_rmspe:.4f}\nStarting to plot {6 * len(self.ugw_n_values)} curves..."
            )

            plot_all_ugw_n_iv_curve_colormap(
                ugw_n_values=self.ugw_n_values,
                plot_data=self.plot_data,
                plot_dir=self.plot_dir,
                # log_y=os.getenv("LOG_Y", "True").lower() == "true",
                log_y=False,
            )
            plot_all_ugw_n_iv_curve_colormap(
                ugw_n_values=self.ugw_n_values,
                plot_data=self.plot_data,
                plot_dir=self.plot_dir,
                log_y=True,
            )
        ### New
        # vto = episode.custom_data["Vto"]
        # avg_vto = np.mean(vto) if vto else 0.0
        # metrics_logger.log_value(
        #     "avg_vto",
        #     avg_vto,
        #     reduce=None,
        # )
        for param_name in tunable_params_names:
            param_values = episode.custom_data[param_name]
            
            avg_param_value = np.mean(param_values) if param_values else 0.0
            
            log_key = f"avg_{param_name}"
            
            metrics_logger.log_value(
                log_key,
                avg_param_value,
                reduce=None,
            )
        return
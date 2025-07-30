import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import verilogae  # Only available on Linux with python 3.11
from dotenv import load_dotenv
from gymnasium import spaces

# Dictionary of all possible tunable parameters
ALL_POSSIBLE_TUNABLE_PARAMS = {
    ## === 臨界電壓相關 ===
    # Vto 預設值: 0.258 。範圍設定涵蓋增強型(E-mode)與空乏型(D-mode)HEMT。
    "Vto": {"min": -1.0, "max": 1.5, "step": 0.01},
    # Gamma 預設值: 0.0095 。通常為一個小的正值。
    "Gamma": {"min": 0.0, "max": 0.3, "step": 0.001},
    # Vch 預設值: 1.4 。此為影響臨界電壓的參數之一。
    "Vch": {"min": 0.5, "max": 3.0, "step": 0.02},
    ## === 跨導與電流增益 ===
    # Gmmax 預設值: 0.168 。範圍涵蓋了典型的RF/功率元件。
    "Gmmax": {"min": 0.05, "max": 0.5, "step": 0.002},
    # Deltgm 預設值: 0.252 。此為跨導的修正因子。
    "Deltgm": {"min": 0.0, "max": 1.0, "step": 0.01},
    ## === 飽和區效應 ===
    # Vsat 預設值: 0.57 。決定I-V曲線膝點(knee)電壓，通常在1V上下。
    "Vsat": {"min": 0.1, "max": 2.0, "step": 0.01},
    # Kapa 預設值: 0.069 。功能同通道長度調變 Lambda，值通常較小。
    "Kapa": {"min": 0.0, "max": 0.3, "step": 0.001},
    # Alpha 預設值: 0.01 。作為轉態區的平滑化因子，通常為一小正數。
    "Alpha": {"min": 0.001, "max": 0.2, "step": 0.001},
    # Peff 預設值: 1.53 。與自熱效應相關，範圍可較大。
    "Peff": {"min": 0.5, "max": 10.0, "step": 0.05},
    ## === 寄生電阻 ===
    # Rs 預設值: 2.0 。範圍涵蓋小訊號到功率元件的典型值。
    "Rs": {"min": 0.1, "max": 10.0, "step": 0.1},
    # Rd 預設值: 1.0 。範圍涵蓋小訊號到功率元件的典型值。
    "Rd": {"min": 0.1, "max": 10.0, "step": 0.1},
}

# Get tunable params name from environment variable
load_dotenv()
tunable_param_names = [
    name.strip() for name in os.getenv("TUNABLE_PARAMS", "").split(",") if name.strip()
]

# Set tunable params config
tunable_params_config = {}
for name in tunable_param_names:
    if name in ALL_POSSIBLE_TUNABLE_PARAMS:
        tunable_params_config[name] = ALL_POSSIBLE_TUNABLE_PARAMS[name]
    else:
        print(
            f"Warning: Parameter '{name}' from environment variable not found in master config. Skipping."
        )


class EEHEMTEnv(gym.Env):
    """
    A custom Gymnasium environment for optimizing EE-HEMT model parameters.

    Attributes:
        action_space (gym.spaces.Box): The space of possible actions.
        observation_space (gym.spaces.Box): The space of possible observations.
        ...
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        """
        Initializes the environment.

        Args:
            config (dict): A dictionary containing configuration parameters for the environment,
                           such as file paths and parameter tuning settings.
        """
        super(EEHEMTEnv, self).__init__()

        self.eehemt_model = verilogae.load(config.get("va_file_path", ""))
        self.temperature = config.get("temperature", 300.0)
        self.csv_file_path = config.get("csv_file_path", "")

        # Initialize All Params
        self.tunable_params_config = config.get("tunable_params_config", {})
        self.tunable_param_names = list(self.tunable_params_config.keys())
        
        self.initial_params = {
            name: param.default for name, param in self.eehemt_model.modelcard.items()
        }
        self.current_params = self.initial_params.copy()
        test_modified = config.get("test_modified", False)
        if test_modified:
            self.modified_initial_params = self.initial_params.copy()
            for name in self.tunable_param_names:
                self.modified_initial_params[name] *= 1.5

        # Load measured data and set up sweep bias
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(
                f"Measured data file not found:: {self.csv_file_path}"
            )
        measured_data = pd.read_csv(self.csv_file_path)
        self.vgs = measured_data["vg"].values
        vds = np.full_like(self.vgs, 0.1)
        self.sweep_bias = {
            "br_gisi": self.vgs,
            "br_disi": vds,
            "br_t": self.vgs,
            "br_esi": self.vgs,
        }
        
        if test_modified:
            self.i_meas = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.modified_initial_params,
            )
        else:
            #$ 原本用真正的 csv file 的 id 做 i_meas
            self.i_meas = measured_data["id"].values

        # Define Action Space
        action_deltas = np.array(
            [config["step"] for config in self.tunable_params_config.values()],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-action_deltas, high=action_deltas, dtype=np.float32
        )

        # Define Observation Space
        param_mins = [config["min"] for config in self.tunable_params_config.values()]
        param_maxs = [config["max"] for config in self.tunable_params_config.values()]

        low_bounds = np.append(param_mins, -np.inf).astype(np.float32)
        high_bounds = np.append(param_maxs, np.inf).astype(np.float32)
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, dtype=np.float32
        )

        self.initial_rmspe = -1.0
        self.previous_rmspe = -1.0

        self._running_stats_count = 1e-4
        self._running_stats_mean = 0.0
        self._running_stats_M2 = 0.0

        self.max_episode_steps = 10000
        self.rmspe_threshold = 0.05
        self.current_step = 0

    def _calculate_rmspe(self, i_sim: np.ndarray) -> float:
        """
        Calculates the Root Mean Square Percentage Error (RMSPE) between
        simulated current and measured current.

        Args:
            i_sim (np.ndarray): The array of simulated current values.

        Returns:
            float: The calculated RMSPE value.
        """
        i_meas_safe = np.where(np.abs(self.i_meas) < 1e-12, 1e-12, self.i_meas)
        rmspe = np.sqrt(np.mean(np.square((self.i_meas - i_sim) / i_meas_safe)))
        return rmspe

    def _update_running_stats(self, new_rmspe_value: float):
        """
        Updates the running mean and variance of the RMSPE using Welford's online algorithm.
        This is used for observation normalization.

        Args:
            new_rmspe_value (float): The new RMSPE value to incorporate into the stats.
        """
        self._running_stats_count += 1
        delta = new_rmspe_value - self._running_stats_mean
        self._running_stats_mean += delta / self._running_stats_count
        delta2 = new_rmspe_value - self._running_stats_mean
        self._running_stats_M2 += delta * delta2

    def _get_obs(self, rmspe: float) -> np.ndarray:
        """
        Constructs the observation array for the agent.

        The observation consists of the current tunable parameter values and the
        normalized RMSPE. Normalization is done using running statistics.

        Args:
            rmspe (float): The current RMSPE value.

        Returns:
            np.ndarray: The observation array.
        """
        self._update_running_stats(rmspe)
        running_var = self._running_stats_M2 / self._running_stats_count
        running_std = np.sqrt(running_var)
        normalized_rmspe = (rmspe - self._running_stats_mean) / (running_std + 1e-8)

        param_values = np.array(
            [self.current_params[name] for name in self.tunable_param_names],
            dtype=np.float32,
        )
        return np.append(param_values, normalized_rmspe).astype(np.float32)

    def _get_info(self, rmspe: float) -> dict:
        """
        Generates the info dictionary returned at each step.

        Args:
            rmspe (float): The current RMSPE value.

        Returns:
            dict: A dictionary containing auxiliary diagnostic information.
        """
        return {"current_rmspe": rmspe, "current_params": self.current_params}

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """
        Resets the environment to its initial state for a new episode.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and info dictionary.
        """
        super().reset(seed=seed)
        self.current_params = self.initial_params.copy()
        self.current_step = 0

        initial_i_sim = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **self.current_params,
        )
        self.initial_rmspe = self._calculate_rmspe(initial_i_sim)
        self.previous_rmspe = self.initial_rmspe

        print(
            f"Initial Params (tunable part): {{ {', '.join(f'{k}: {v:.4f}' for k, v in {name: self.current_params[name] for name in self.tunable_param_names}.items())} }}"
        )
        print(f"Initial RMSPE: {self.initial_rmspe:.4f}")

        observation = self._get_obs(self.initial_rmspe)
        info = self._get_info(self.initial_rmspe)

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        """
        Executes one time step within the environment.

        This involves updating the model parameters based on the agent's action,
        simulating the I-V curve, calculating the new RMSPE, and determining the reward.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            tuple: A tuple containing the new observation, reward, terminated flag,
                   truncated flag, and info dictionary.
        """
        self.current_step += 1

        for i, param_name in enumerate(self.tunable_param_names):
            self.current_params[param_name] += action[i]
            min_val = self.tunable_params_config[param_name]["min"]
            max_val = self.tunable_params_config[param_name]["max"]
            self.current_params[param_name] = np.clip(
                self.current_params[param_name], min_val, max_val
            )

        i_sim = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **self.current_params,
        )

        current_rmspe = self._calculate_rmspe(i_sim)
        reward = self.previous_rmspe - current_rmspe
        self.previous_rmspe = current_rmspe

        terminated = current_rmspe < self.rmspe_threshold
        truncated = self.current_step >= self.max_episode_steps

        if terminated:
            print(
                f"Success! RMSPE ({current_rmspe:.4f}) has reached the threshold ({self.rmspe_threshold})."
            )
        if truncated and not terminated:
            print("Reached maximum steps.")

        observation = self._get_obs(current_rmspe)
        info = self._get_info(current_rmspe)

        return observation, reward, terminated, truncated, info

    def plot_iv_curve(
        self,
        plot_initial: bool = True,
        plot_modified: bool = True,
        plot_current: bool = True,
        save_path: str = None,
    ):
        """
        Plots and compares I-V curves for different parameter sets.

        Args:
            plot_initial (bool): Whether to plot the curve using self.initial_params.
            plot_modified (bool): Whether to plot the curve using self.modified_initial_params.
            plot_current (bool): Whether to plot the curve using self.current_params (optimized by the agent).
            save_path (str, optional): If provided, the plot will be saved to this path instead of being displayed.
        """
        plt.figure(figsize=(10, 7))

        # Plot measured data as a baseline
        plt.plot(self.vgs, self.i_meas, "ko", label="Measured Data (Target)")

        # Plot different simulated curves based on options
        if plot_initial:
            i_sim_initial = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.initial_params,
            )
            plt.plot(self.vgs, i_sim_initial, "b--", label="Simulated (Initial Params)")

        if plot_modified:
            i_sim_modified = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.modified_initial_params,
            )
            plt.plot(
                self.vgs,
                i_sim_modified,
                "g-.",
                label="Simulated (Modified Initial Params)",
            )

        if plot_current:
            i_sim_current = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.current_params,
            )
            plt.plot(self.vgs, i_sim_current, "r-", label="Simulated (Current Params)")

        # Style the plot
        plt.title("I-V Curve Comparison")
        plt.xlabel("Gate Voltage (Vg) [V]")
        plt.ylabel("Drain Current (Id) [A]")
        plt.yscale("log")  # Log scale is often used for I-V curves
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"I-V curve plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

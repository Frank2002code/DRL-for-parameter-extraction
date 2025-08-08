import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Only available on Linux with python 3.11
import verilogae  # type: ignore[import-untyped]
from dotenv import load_dotenv
from gymnasium.spaces import Box

from utils.metrics import calculate_rmspe

# Dictionary of all possible tunable parameters
ALL_POSSIBLE_TUNABLE_PARAMS = {
    ## === 臨界電壓相關 ===
    # Vto 預設值: 0.258 。範圍設定涵蓋增強型(E-mode)與空乏型(D-mode)HEMT。
    "Vto": {"min": -1.0, "max": 1.5, "factor": 0.01},
    # Vtso 預設值: 0.358 。為飽和區的臨界電壓參數。
    "Vtso": {"min": 0.0, "max": 1.5, "factor": 0.01},
    # Vgo 預設值: 0.618 。為跨導模型中的閘極電壓參數。
    "Vgo": {"min": 0.0, "max": 1.5, "factor": 0.01},
    # Vco 預設值: 0.75 。為輸出電導模型中的交叉電壓。
    "Vco": {"min": 0.0, "max": 2.0, "factor": 0.01},
    # Vch 預設值: 1.4 。此為影響臨界電壓的參數之一。
    "Vch": {"min": 0.5, "max": 3.0, "factor": 0.02},
    # Gamma 預設值: 0.0095 。通常為一個小的正值。
    "Gamma": {"min": 0.0, "max": 0.3, "factor": 0.001},
    ## === 跨導與電流增益 ===
    # Gmmax 預設值: 0.168 。範圍涵蓋了典型的RF/功率元件。
    "Gmmax": {"min": 0.05, "max": 0.5, "factor": 0.002},
    # Deltgm 預設值: 0.252 。此為跨導的修正因子。
    "Deltgm": {"min": 0.0, "max": 1.0, "factor": 0.01},
    ## === 飽和區效應 ===
    # Vsat 預設值: 0.57 。決定I-V曲線膝點(knee)電壓，通常在1V上下。
    "Vsat": {"min": 0.1, "max": 2.0, "factor": 0.01},
    # Kapa 預設值: 0.069 。功能同通道長度調變 Lambda，值通常較小。
    "Kapa": {"min": 0.0, "max": 0.3, "factor": 0.001},
    # Peff 預設值: 1.53 。與自熱效應相關，範圍可較大。
    "Peff": {"min": 0.5, "max": 10.0, "factor": 0.05},
    ## === 二階效應 ===
    # Alpha 預設值: 0.01 。作為轉態區的平滑化因子，通常為一小正數。
    "Alpha": {"min": 0.001, "max": 0.2, "factor": 0.001},
    # Mu 預設值: 7.86e-6 。為遷移率退化係數。
    "Mu": {"min": 1e-7, "max": 1e-4, "factor": 1e-7},
    # Vbc 預設值: 0.95 。為崩潰電壓相關參數。
    "Vbc": {"min": 0.1, "max": 5.0, "factor": 0.05},
    ## === 寄生電阻 ===
    # Rs 預設值: 2.0 。範圍涵蓋小訊號到功率元件的典型值。
    "Rs": {"min": 0.1, "max": 10.0, "factor": 0.1},
    # Rd 預設值: 1.0 。範圍涵蓋小訊號到功率元件的典型值。
    "Rd": {"min": 0.1, "max": 10.0, "factor": 0.1},
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
### New
EPSILON = 1e-9


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

        self.init_params = {
            name: param.default for name, param in self.eehemt_model.modelcard.items()
        }
        self.test_modified = config.get("test_modified", False)
        if self.test_modified:
            self.modified_init_params = self.init_params.copy()
            for name in self.tunable_param_names:
                self.modified_init_params[name] *= 1.2
            self.current_params = self.modified_init_params.copy()
        else:
            self.current_params = self.init_params.copy()

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

        if self.test_modified:
            self.i_meas = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.init_params,
            )
        else:
            # $ 原本用真正的 csv file 的 id 做 i_meas
            self.i_meas = measured_data["id"].values

        # Define Action Space
        action_deltas = np.array(
            [config["factor"] for config in self.tunable_params_config.values()],
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-action_deltas, high=action_deltas, dtype=np.float32
        )

        # Define Observation Space
        param_mins = [config["min"] for config in self.tunable_params_config.values()]
        param_maxs = [config["max"] for config in self.tunable_params_config.values()]

        low_bounds = np.append(param_mins, -np.inf).astype(np.float32)
        high_bounds = np.append(param_maxs, np.inf).astype(np.float32)
        self.observation_space = Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        self.init_rmspe = -1.0
        self.prev_rmspe = -1.0

        self._running_stats_count = 1e-4
        self._running_stats_mean = 0.0
        self._running_stats_M2 = 0.0

        self.max_episode_steps = 2000
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

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        """
        Resets the environment to its initial state for a new episode.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and info dictionary.
        """
        super().reset(seed=seed)
        if self.test_modified:
            self.current_params = self.modified_init_params.copy()
        else:
            self.current_params = self.init_params.copy()
        self.current_step = 0

        initial_i_sim = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **self.current_params,
        )
        self.init_rmspe = calculate_rmspe(self.i_meas, initial_i_sim)
        self.prev_rmspe = self.init_rmspe

        # print(
        #     f"Initial Params (tunable part): {{ {', '.join(f'{k}: {v:.4f}' for k, v in {name: self.current_params[name] for name in self.tunable_param_names}.items())} }}"
        # )
        # print(f"Initial RMSPE: {self.init_rmspe:.4f}")

        observation = self._get_obs(self.init_rmspe)
        info = self._get_info(self.init_rmspe)

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

        params_for_eval = {k: float(v) for k, v in self.current_params.items()}
        i_sim = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **params_for_eval,
        )

        current_rmspe = calculate_rmspe(self.i_meas, i_sim)
        reward = self.prev_rmspe - current_rmspe
        self.prev_rmspe = current_rmspe

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
        save_path: str | None = None,
    ):
        """
        Plots and compares I-V curves for different parameter sets.

        Args:
            plot_initial (bool): Whether to plot the curve using self.init_params.
            plot_modified (bool): Whether to plot the curve using self.modified_init_params.
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
                **self.init_params,
            )
            plt.plot(self.vgs, i_sim_initial, "b--", label="Simulated (Initial Params)")

        if plot_modified:
            i_sim_modified = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.modified_init_params,
            )
            plt.plot(
                self.vgs,
                i_sim_modified,
                "g-.",
                label="Simulated (Modified Initial Params)",
            )

        if plot_current:
            params_for_eval = {k: float(v) for k, v in self.current_params.items()}
            i_sim_current = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **params_for_eval,
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


class EEHEMTEnv_Norm(gym.Env):
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
        super(EEHEMTEnv_Norm, self).__init__()

        self.eehemt_model = verilogae.load(config.get("va_file_path", ""))
        self.temperature = config.get("temperature", 300.0)
        self.csv_file_path = config.get("csv_file_path", "")

        # === All Params (Including Tunable) Initialization ===
        self.tunable_params_config = config.get("tunable_params_config", {})
        self.tunable_param_names = list(self.tunable_params_config.keys())
        self.test_modified = config.get("test_modified", False)

        self.init_params = {
            name: param.default for name, param in self.eehemt_model.modelcard.items()
        }
        if self.test_modified:
            self.modified_init_params = self.init_params.copy()
            for name in self.tunable_param_names:
                self.modified_init_params[name] *= 1.2
            self.current_params = self.modified_init_params.copy()
        else:
            self.current_params = self.init_params.copy()
        ### New
        self.current_tunable_params = np.array(
            [self.current_params[name] for name in self.tunable_param_names],
            dtype=np.float32,
        )
        self.TUNABLE_PARAMS_MIN = np.array(
            [config["min"] for config in self.tunable_params_config.values()],
            dtype=np.float32,
        )
        self.TUNABLE_PARAMS_MAX = np.array(
            [config["max"] for config in self.tunable_params_config.values()],
            dtype=np.float32,
        )

        # === Load I_meas (y_true) and sweep bias ===
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
        if self.test_modified:
            self.i_meas = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.init_params,
            )
        else:
            # $ 原本用真正的 csv file 的 id 做 i_meas
            self.i_meas = measured_data["id"].values

        # === Action Space Definition ===
        self.action_space = Box(low=-1.0, high=1.0, dtype=np.float32)
        self.ACTION_FACTORS = np.array(
            [config["factor"] for config in self.tunable_params_config.values()],
            dtype=np.float32,
        )  # Linear transform better than independent function transform
        self.prev_params_delta = np.zeros_like(self.current_tunable_params)

        # === Observation Space Definition ===
        # Observation space contains: [P_t, ΔP_{t-1}, E_t (raw error)]
        param_low = [config["min"] for config in self.tunable_params_config.values()]
        param_high = [config["max"] for config in self.tunable_params_config.values()]

        prev_params_delta_low = -self.ACTION_FACTORS
        prev_params_delta_high = self.ACTION_FACTORS

        err_vector_low = np.full(len(self.i_meas), -np.inf)
        err_vector_high = np.full(len(self.i_meas), np.inf)

        low_bounds = np.concatenate(
            [param_low, prev_params_delta_low, err_vector_low]
        ).astype(np.float32)
        high_bounds = np.concatenate(
            [param_high, prev_params_delta_high, err_vector_high]
        ).astype(np.float32)
        self.observation_space = Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # === Episode Control ===
        self.MAX_EPISODE_STEPS = int(os.getenv("MAX_EPISODE_STEPS", 1000))
        self.RMSPE_THRESHOLD = float(os.getenv("RMSPE_THRESHOLD", 0.05))
        self.current_step = 0

        # === Error Initialization ===
        self.init_rmspe = -1.0
        self.prev_rmspe = -1.0  # For reward calculation

        ### New
        # === Stagnation (停滯) detection settings ===
        self.STAGNATION_PATIENCE_STEPS = int(
            os.getenv("STAGNATION_PATIENCE_STEPS", 50)
        )  # step 耐心值
        self.STAGNATION_THRESHOLD = float(
            os.getenv("STAGNATION_THRESHOLD", 1e-6)
        )  # 進展的門檻
        self.stagnation_cnt = 0

    def _get_obs(self, i_sim: np.ndarray) -> np.ndarray:
        """
        Constructs the observation vector for the agent.

        Observation = [P_t (current params), ΔP_{t-1} (previous param change), E_t (normalized error vector)]

        Returns:
            np.ndarray: The observation vector.
        """
        # 1. P_t: Current tunable params vector

        # 2. \Delta_P_{t-1}: Diff vector between current and previous params

        # 3. E_t: Error vector between I_meas and I_sim
        err_vector = self.i_meas - i_sim

        # 4. Combine observation vector
        obs = np.concatenate(
            [self.current_tunable_params, self.prev_params_delta, err_vector]
        ).astype(np.float32)

        return obs

    def _get_info(self, rmspe: float) -> dict:
        """
        Generates the info dictionary returned at each step.

        Args:
            rmspe (float): The current RMSPE value.

        Returns:
            dict: A dictionary containing auxiliary diagnostic information.
        """
        return {"current_rmspe": rmspe, "current_params": self.current_params.copy()}

    ### New
    def _transform_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse transform function: converts normalized action [-1, 1] to actual parameter changes."""
        return action * self.ACTION_FACTORS

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        """
        Resets the environment to its initial state for a new episode.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and info dictionary.
        """
        super().reset(seed=seed)
        if self.test_modified:
            self.current_params = self.modified_init_params.copy()
        else:
            self.current_params = self.init_params.copy()
        ### New
        self.current_tunable_params = np.array(
            [self.current_params[name] for name in self.tunable_param_names],
            dtype=np.float32,
        )
        self.prev_params_delta = np.zeros_like(self.current_tunable_params)

        self.current_step = 0
        ### New
        self.stagnation_cnt = 0

        # Calculate RMSPE for reward and info
        init_i_sim = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **self.current_params,
        )
        self.init_rmspe = calculate_rmspe(self.i_meas, init_i_sim)
        self.prev_rmspe = self.init_rmspe

        observation = self._get_obs(init_i_sim)
        info = self._get_info(self.init_rmspe)

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

        ### New
        # === Update parameters and ensure they are within defined bounds ===
        tunable_params_delta = self.prev_params_delta = self._transform_action(action)
        self.current_tunable_params += tunable_params_delta
        self.current_tunable_params = np.clip(
            self.current_tunable_params,
            self.TUNABLE_PARAMS_MIN,
            self.TUNABLE_PARAMS_MAX,
        )
        updated_tunable_part = dict(
            zip(self.tunable_param_names, self.current_tunable_params)
        )
        self.current_params.update(updated_tunable_part)

        current_params_float = {
            k: float(v) for k, v in self.current_params.items()
        }  # np.float32 -> float
        i_sim = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **current_params_float,
        )

        # === Calculate RMSPE for reward, termination conditions, and info ===
        current_rmspe = calculate_rmspe(self.i_meas, i_sim)
        reward = self.prev_rmspe - current_rmspe
        self.prev_rmspe = current_rmspe

        # === Get the next observation and info ===
        observation = self._get_obs(i_sim)
        info = self._get_info(current_rmspe)

        # === Check Termination Conditions ===
        terminated_success = current_rmspe < self.RMSPE_THRESHOLD
        if abs(reward) < self.STAGNATION_THRESHOLD:
            self.stagnation_cnt += 1
        else:
            self.stagnation_cnt = 0
        terminated_stagnation = self.stagnation_cnt >= self.STAGNATION_PATIENCE_STEPS
        terminated = terminated_success or terminated_stagnation
        truncated = self.current_step >= self.MAX_EPISODE_STEPS

        if terminated_success:
            print(
                f"Success! RMSPE ({current_rmspe:.4f}) has reached the threshold ({self.RMSPE_THRESHOLD})."
            )
        if terminated_stagnation:
            print(
                f"Terminated due to stagnation ({self.STAGNATION_PATIENCE_STEPS} steps with little improvement)."
            )
        if truncated and not terminated:
            print("Reached maximum steps.")

        if terminated or truncated:
            info["final_rmspe"] = current_rmspe
            # info["final_params"] = self.current_params
            info["plot_data"] = self._get_plot_data()

        return observation, reward, terminated, truncated, info

    def plot_iv_curve(
        self,
        plot_initial: bool = True,
        plot_modified: bool = True,
        plot_current: bool = True,
        save_path: str | None = None,
    ):
        """
        Plots and compares I-V curves for different parameter sets.

        Args:
            plot_initial (bool): Whether to plot the curve using self.init_params.
            plot_modified (bool): Whether to plot the curve using self.modified_init_params.
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
                **self.init_params,
            )
            plt.plot(self.vgs, i_sim_initial, "b--", label="Simulated (Initial Params)")

        if plot_modified:
            i_sim_modified = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.modified_init_params,
            )
            plt.plot(
                self.vgs,
                i_sim_modified,
                "g-.",
                label="Simulated (Modified Initial Params)",
            )

        if plot_current:
            current_params_float = {k: float(v) for k, v in self.current_params.items()}
            i_sim_current = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **current_params_float,
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

    def _get_plot_data(self):
        # 模擬 Initial 曲線
        i_sim_initial = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **self.init_params,
        )
        # 模擬 Modified 曲線
        i_sim_modified = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **self.modified_init_params,
        )
        # 模擬 Current 曲線
        current_params_float = {k: float(v) for k, v in self.current_params.items()}
        i_sim_current = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **current_params_float,
        )

        # 返回一個只包含可序列化數據 (Numpy 陣列) 的字典
        return {
            "vgs": self.vgs,
            "i_meas": self.i_meas,
            "i_sim_initial": i_sim_initial,
            "i_sim_modified": i_sim_modified,
            "i_sim_current": i_sim_current,
        }


class EEHEMTEnv_Norm_Vtos(gym.Env):
    """
    A custom Gymnasium environment for optimizing EE-HEMT model parameters.

    Attributes:
        action_space (gym.spaces.Box): The space of possible actions.
        observation_space (gym.spaces.Box): The space of possible observations.
        ...
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        """
        Initializes the environment.

        Args:
            config (dict): A dictionary containing configuration parameters for the environment,
                           such as file paths and parameter tuning settings.
        """
        super(EEHEMTEnv_Norm_Vtos, self).__init__()

        self.eehemt_model = verilogae.load(config.get("va_file_path", ""))
        self.temperature = config.get("temperature", 300.0)
        self.csv_file_path = config.get("csv_file_path", "")

        ### New
        self.vto_values = [float(v) for v in os.getenv("VTO_VALUES", "-1.0,-0.72,-0.44,-0.16,0.11").split(",")]
        self.num_vtos = len(self.vto_values)

        # === All Params (Including Tunable) Initialization ===
        self.tunable_params_config = config.get("tunable_params_config", {})
        ### New
        if "Vto" in self.tunable_params_config:
            self.tunable_params_config.pop("Vto")
        self.tunable_param_names = list(self.tunable_params_config.keys())
        self.test_modified = config.get("test_modified", False)

        self.init_params = {
            name: param.default for name, param in self.eehemt_model.modelcard.items()
        }
        ### New
        self.init_params["Vto"] = self.vto_values[0]  # Set default Vto

        ### New
        INIT_PARAMS_SHIFT_FACTOR = float(os.getenv("INIT_PARAMS_SHIFT_FACTOR", 1.2))
        if self.test_modified:
            self.modified_init_params = self.init_params.copy()
            for name in self.tunable_param_names:
                self.modified_init_params[name] *= INIT_PARAMS_SHIFT_FACTOR
            self.current_params = self.modified_init_params.copy()
        else:
            self.current_params = self.init_params.copy()

        self.current_tunable_params = np.array(
            [self.current_params[name] for name in self.tunable_param_names],
            dtype=np.float32,
        )
        self.TUNABLE_PARAMS_MIN = np.array(
            [config["min"] for config in self.tunable_params_config.values()],
            dtype=np.float32,
        )
        self.TUNABLE_PARAMS_MAX = np.array(
            [config["max"] for config in self.tunable_params_config.values()],
            dtype=np.float32,
        )

        # === Load I_meas (y_true) and sweep bias ===
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
        if self.test_modified:
            # self.i_meas = self.eehemt_model.functions["Ids"].eval(
            #     temperature=self.temperature,
            #     voltages=self.sweep_bias,
            #     **self.init_params,
            # )
            self.i_meas_dict = {
                vto: self.eehemt_model.functions["Ids"].eval(
                    temperature=self.temperature,
                    voltages=self.sweep_bias,
                    **(self.init_params | {"Vto": float(vto)}),
                )
                for vto in self.vto_values
            }
            print("\nSynthetic target data generation complete.\n")
        else:
            # $ 原本用真正的 csv file 的 id 做 i_meas
            # self.i_meas_dict = measured_data["id"].values
            pass

        # === Action Space Definition ===
        self.action_space = Box(low=-1.0, high=1.0, dtype=np.float32)
        self.ACTION_FACTORS = np.array(
            [config["factor"] for config in self.tunable_params_config.values()],
            dtype=np.float32,
        )  # Linear transform better than independent function transform
        self.prev_params_delta = np.zeros_like(self.current_tunable_params)

        # === Observation Space Definition ===
        # Observation space contains: [P_t, ΔP_{t-1}, E_t (raw error)]
        param_low = [config["min"] for config in self.tunable_params_config.values()]
        param_high = [config["max"] for config in self.tunable_params_config.values()]

        prev_params_delta_low = -self.ACTION_FACTORS
        prev_params_delta_high = self.ACTION_FACTORS

        total_err_len = len(self.vgs) * self.num_vtos
        err_vector_low = np.full(total_err_len, -np.inf)
        err_vector_high = np.full(total_err_len, np.inf)

        low_bounds = np.concatenate(
            [param_low, prev_params_delta_low, err_vector_low]
        ).astype(np.float32)
        high_bounds = np.concatenate(
            [param_high, prev_params_delta_high, err_vector_high]
        ).astype(np.float32)
        self.observation_space = Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # === Episode Control ===
        self.MAX_EPISODE_STEPS = int(os.getenv("MAX_EPISODE_STEPS", 1000))
        self.RMSPE_THRESHOLD = float(os.getenv("RMSPE_THRESHOLD", 0.15))
        self.current_step = 0

        # === Error Initialization ===
        # self.init_rmspe = -1.0
        self.prev_rmspe = -1.0  # For reward calculation

        # === Stagnation (停滯) detection settings ===
        self.STAGNATION_PATIENCE_STEPS = int(
            os.getenv("STAGNATION_PATIENCE_STEPS", 50)
        )  # step 耐心值
        self.STAGNATION_THRESHOLD = float(
            os.getenv("STAGNATION_THRESHOLD", 1e-3)
        )  # 進展的門檻
        self.stagnation_cnt = 0

    def _get_obs(self, concat_err_vector: np.ndarray) -> np.ndarray:
        """
        Constructs the observation vector for the agent.

        Observation = [P_t (current params), ΔP_{t-1} (previous param change), E_t (normalized error vector)]

        Returns:
            np.ndarray: The observation vector.
        """
        # 1. P_t: Current tunable params vector

        # 2. \Delta_P_{t-1}: Diff vector between current and previous params

        # 3. E_t: Error vector between I_meas and I_sim

        # 4. Combine observation vector
        obs = np.concatenate(
            [self.current_tunable_params, self.prev_params_delta, concat_err_vector]
        ).astype(np.float32)

        return obs

    def _get_info(self, rmspe: float) -> dict:
        """
        Generates the info dictionary returned at each step.

        Args:
            rmspe (float): The current RMSPE value.

        Returns:
            dict: A dictionary containing auxiliary diagnostic information.
        """
        return {"current_rmspe": rmspe, "current_params": self.current_params.copy()}

    def _transform_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse transform function: converts normalized action [-1, 1] to actual parameter changes."""
        return action * self.ACTION_FACTORS

    ### New
    def _run_all_vto_sim(self) -> tuple[np.ndarray, np.ndarray]:
        """
        NEW: Helper function to run simulations for all Vto conditions.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - A flattened numpy array containing all concatenated error vectors.
                - A numpy array of RMSPE values for each Vto condition.
        """
        all_i_meas_matrix = np.array([self.i_meas_dict[vto] for vto in self.vto_values])

        current_params_float = {k: float(v) for k, v in self.current_params.items()}
        all_i_sim_matrix = np.array(
            [
                self.eehemt_model.functions["Ids"].eval(
                    temperature=self.temperature,
                    voltages=self.sweep_bias,
                    # The `|` operator merges the base parameters with the current Vto
                    **(current_params_float | {"Vto": vto}),
                )
                for vto in self.vto_values
            ]
        )

        all_err_matrix = (
            all_i_meas_matrix - all_i_sim_matrix
        )  # self.i_meas - all_i_sim_matrix

        concat_err_vector = all_err_matrix.flatten().astype(np.float32)

        # Calculate RMSPE for each I-V curve (each row).
        rmspe_vals = np.array(
            [
                calculate_rmspe(i_meas_row, i_sim_row)
                for i_meas_row, i_sim_row in zip(all_i_meas_matrix, all_i_sim_matrix)
            ],
            dtype=np.float32,
        )

        return concat_err_vector, rmspe_vals

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        """
        Resets the environment to its initial state for a new episode.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and info dictionary.
        """
        super().reset(seed=seed)
        if self.test_modified:
            self.current_params = self.modified_init_params.copy()
        else:
            self.current_params = self.init_params.copy()
        ### New
        self.current_tunable_params = np.array(
            [self.current_params[name] for name in self.tunable_param_names],
            dtype=np.float32,
        )
        self.prev_params_delta = np.zeros_like(self.current_tunable_params)

        self.current_step = 0
        self.stagnation_cnt = 0

        # === Run initial simulation for all Vto conditions & Calculate RMSPE ===
        # init_i_sim = self.eehemt_model.functions["Ids"].eval(
        #     temperature=self.temperature,
        #     voltages=self.sweep_bias,
        #     **self.current_params,
        # )
        init_err_vector, init_rmspe_vals = self._run_all_vto_sim()
        avg_init_rmspe = np.mean(init_rmspe_vals)
        # self.init_rmspe = avg_init_rmspe
        self.prev_rmspe = avg_init_rmspe

        observation = self._get_obs(init_err_vector)
        info = self._get_info(avg_init_rmspe)

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

        ### New
        # === Update parameters and ensure they are within defined bounds ===
        tunable_params_delta = self.prev_params_delta = self._transform_action(action)
        self.current_tunable_params += tunable_params_delta
        self.current_tunable_params = np.clip(
            self.current_tunable_params,
            self.TUNABLE_PARAMS_MIN,
            self.TUNABLE_PARAMS_MAX,
        )
        updated_tunable_part = dict(
            zip(self.tunable_param_names, self.current_tunable_params)
        )
        self.current_params.update(updated_tunable_part)

        # === Run simulations for all Vto conditions ===
        # current_params_float = {
        #     k: float(v) for k, v in self.current_params.items()
        # }  # np.float32 -> float
        # i_sim = self.eehemt_model.functions["Ids"].eval(
        #     temperature=self.temperature,
        #     voltages=self.sweep_bias,
        #     **current_params_float,
        # )
        current_err_vector, rmspe_vals = self._run_all_vto_sim()

        # === Calculate RMSPE for reward, termination conditions, and info ===
        current_rmspe = np.mean(rmspe_vals)
        ### New
        reward = (self.prev_rmspe - current_rmspe) / (self.prev_rmspe + EPSILON)
        reward = np.clip(reward, -1.0, 1.0)  # Normalize reward to [-1, 1]
        self.prev_rmspe = current_rmspe

        # === Get the next observation and info ===
        observation = self._get_obs(current_err_vector)
        info = self._get_info(current_rmspe)

        # === Check Termination Conditions ===
        terminated_success = current_rmspe < self.RMSPE_THRESHOLD
        if abs(reward) < self.STAGNATION_THRESHOLD:
            self.stagnation_cnt += 1
        else:
            self.stagnation_cnt = 0
        terminated_stagnation = self.stagnation_cnt >= self.STAGNATION_PATIENCE_STEPS
        terminated = terminated_success or terminated_stagnation
        truncated = self.current_step >= self.MAX_EPISODE_STEPS

        if terminated_success:
            print(
                f"Success! RMSPE ({current_rmspe:.4f}) has reached the threshold ({self.RMSPE_THRESHOLD})."
            )
        if terminated_stagnation:
            print(
                f"Terminated due to stagnation ({self.STAGNATION_PATIENCE_STEPS} steps with little improvement)."
            )
        if truncated and not terminated:
            print("Reached maximum steps.")

        if terminated or truncated:
            info["final_rmspe"] = current_rmspe
            # info["final_params"] = self.current_params
            # info["i_sim_current"] = self._get_i_sim_current()
            info["i_sim_current_matrix"] = self._get_i_sim_current_matrix()

        return observation, reward, terminated, truncated, info

    def plot_iv_curve(
        self,
        plot_initial: bool = True,
        plot_modified: bool = True,
        plot_current: bool = True,
        save_path: str | None = None,
    ) -> None:
        """
        Plots and compares I-V curves.
        The target curve is chosen to be the one corresponding to the FIRST Vto value.
        """
        i_meas = self.i_meas_dict[
            self.vto_values[0]
        ]  # Use the first Vto value as target
        plt.figure(figsize=(10, 7))
        plt.plot(self.vgs, i_meas, "ko", label="Measured Data (Target)")

        if plot_initial:
            i_sim_initial = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.init_params,
            )
            plt.plot(
                self.vgs,
                i_sim_initial,
                "b--",
                label=f"Simulated (Initial, Vto={self.init_params.get('Vto'):.2f})",
            )

        if plot_modified and self.test_modified:
            i_sim_modified = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.modified_init_params,
            )
            plt.plot(
                self.vgs,
                i_sim_modified,
                "g-.",
                label=f"Simulated (Modified, Vto={self.modified_init_params.get('Vto'):.2f})",
            )

        if plot_current:
            current_vto = self.current_params.get("Vto")
            current_params_float = {k: float(v) for k, v in self.current_params.items()}
            i_sim_current = self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **current_params_float,
            )
            plt.plot(
                self.vgs,
                i_sim_current,
                "r-",
                label=f"Simulated (Current, Vto={current_vto:.2f})",
            )

        plt.title("I-V Curve Comparison")
        plt.xlabel("Gate Voltage (Vg) [V]")
        plt.ylabel("Drain Current (Id) [A]")
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

    def _get_plot_data(self):
        first_vto = self.vto_values[0]
        i_meas = self.i_meas_dict[first_vto]

        # Simulate Initial Curve
        i_sim_initial = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **self.init_params,
        )
        # Simulate Modified Curve
        i_sim_modified = (
            self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **self.modified_init_params,
            )
            if self.test_modified
            else np.zeros_like(self.vgs)
        )
        # Simulate Current Curve
        # current_params_float = {k: float(v) for k, v in self.current_params.items()}
        # i_sim_current = self.eehemt_model.functions["Ids"].eval(
        #     temperature=self.temperature,
        #     voltages=self.sweep_bias,
        #     **current_params_float,
        # )

        return {
            "vgs": self.vgs,
            "i_meas": i_meas,
            "i_sim_initial": i_sim_initial,
            "i_sim_modified": i_sim_modified,
            # "i_sim_current": i_sim_current,
            ### New
            "vto": first_vto,
        }

    ### New
    def _get_plot_data_matrix(self):
        i_sim_initial_matrix = np.array([
            self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **(self.init_params | {"Vto": float(vto)})
            ) for vto in self.vto_values
        ])
        i_sim_modified_matrix = np.array([
            self.eehemt_model.functions["Ids"].eval(
                temperature=self.temperature,
                voltages=self.sweep_bias,
                **(self.modified_init_params | {"Vto": float(vto)})
            ) for vto in self.vto_values
        ]) if self.test_modified else np.zeros((len(self.vto_values), len(self.vgs)))

        return {
            "vgs": self.vgs,
            "i_meas_dict": self.i_meas_dict,
            "i_sim_initial_matrix": i_sim_initial_matrix,
            "i_sim_modified_matrix": i_sim_modified_matrix,
        }

    ### New
    def _get_i_sim_current(self) -> dict:
        """
        Returns only the DYNAMIC data for plotting at the end of an episode.
        """
        # Simulate Current Curve
        current_params_float = {k: float(v) for k, v in self.current_params.items()}
        i_sim_current = self.eehemt_model.functions["Ids"].eval(
            temperature=self.temperature,
            voltages=self.sweep_bias,
            **current_params_float,
        )

        return {"i_sim_current": i_sim_current}

    def _get_i_sim_current_matrix(self) -> dict[str, np.ndarray]:
        """
        Returns the DYNAMIC data matrix for plotting at the end of an episode.
        """
        current_params_float = {k: float(v) for k, v in self.current_params.items()}
        i_sim_current_matrix = np.array(
            [
                self.eehemt_model.functions["Ids"].eval(
                    temperature=self.temperature,
                    voltages=self.sweep_bias,
                    # The `|` operator merges the base parameters with the current Vto
                    **(current_params_float | {"Vto": vto}),
                )
                for vto in self.vto_values
            ]
        )

        return {"i_sim_current_matrix": i_sim_current_matrix}
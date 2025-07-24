import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import verilogae  # Only available on Linux with python 3.11

class EEHEMTEnv(gym.Env):
    """
    一個客製化的 Gymnasium 環境，用於優化 EE-HEMT 模型參數

    Attributes:
        action_space (gym.spaces.Box): 定義了每個可調參數的允許變化範圍。
        observation_space (gym.spaces.Box): 定義了狀態空間，包括參數值和 RMSE。
        ...
    """
    metadata = {'render_modes': []}

    def __init__(self, va_file, csv_file, tunable_params_config):
        """
        初始化環境。

        Args:
            va_file (str): Verilog-A 模型檔案的路徑 (.va)。
            csv_file (str): 包含量測數據的 CSV 檔案路徑。
            tunable_params_config (dict): 一個描述可調整參數、初始值和邊界的字典。
                                          格式: {'param_name': {'initial': float, 'min': float, 'max': float, 'delta': float}}
        """
        super(EEHEMTEnv, self).__init__()

        # self.va_file_path = va_file
        self.csv_file_path = csv_file
        self.params_config = tunable_params_config
        self.tunable_param_names = list(tunable_params_config.keys())
        self.eehemt_model = verilogae.load(va_file)
        self.temperature = 300
        vgs = np.linspace(0, 1, 101)
        vds = np.full_like(vgs, 0.1)
        self.sweep_bias = {
            'br_gsi': vgs,
            'br_disi': vds,
            'br_t': vgs,
            'br_esi': vgs
        }

        # 1. 載入真實量測數據 I_meas
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Measured data file not found:: {csv_file}")
        measured_data = pd.read_csv(self.csv_file_path)
        self.vg_values = measured_data['vg'].values
        self.i_meas = measured_data['id'].values

        # 2. Define Action Space
        # Action 是對每個參數的 "變化量"。範圍是 [-delta, +delta]。
        action_deltas = np.array([config['delta'] for config in self.params_config.values()], dtype=np.float32)
        self.action_space = spaces.Box(low=-action_deltas, high=action_deltas, dtype=np.float32)

        # 3. Define Observation Space
        # 狀態 = [param1_val, param2_val, ..., rmse]
        param_mins = [config['min'] for config in self.params_config.values()]
        param_maxs = [config['max'] for config in self.params_config.values()]
        
        # 狀態的邊界
        # 參數值的邊界 + RMSE 的邊界 (0 到 1，因為我們會用正規化的 RMSE)
        low_bounds = np.append(param_mins, -np.inf).astype(np.float32)
        high_bounds = np.append(param_maxs, np.inf).astype(np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        
        # 4. 初始化內部狀態
        self.initial_params = {name: config['initial'] for name, config in self.params_config.items()}
        self.current_params = self.initial_params.copy()
        self.initial_rmse = -1.0
        self.previous_rmse = -1.0
        
        self._running_stats_count = 1e-4 # 使用一個極小值來避免初始除以零
        self._running_stats_mean = 0.0
        self._running_stats_M2 = 0.0 # M2 用於計算變異數
        
        # 設定終止條件
        self.max_episode_steps = 100 # 每回合最大步數
        self.rmse_threshold = 0.05   # RMSE 低於 5% 時，視為成功
        self.current_step = 0

    def _calculate_rmse(self, i_sim):
        """計算 Relative RMSE (%)，與您在 notebook 中的算法保持一致。"""
        # 避免除以零
        i_meas_safe = np.where(np.abs(self.i_meas) < 1e-12, 1e-12, self.i_meas)
        # RMSPE: Root Mean Square Percentage Error
        rmspe = np.sqrt(np.mean(np.square((self.i_meas - i_sim) / i_meas_safe)))
        return rmspe
    
    def _update_running_stats(self, new_rmse_value: float):
        """Update running statistics for RMSE."""
        self._running_stats_count += 1
        delta = new_rmse_value - self._running_stats_mean
        self._running_stats_mean += delta / self._running_stats_count
        delta2 = new_rmse_value - self._running_stats_mean
        self._running_stats_M2 += delta * delta2

    def _get_obs(self, rmse: float):
        self._update_running_stats(rmse)

        running_var = self._running_stats_M2 / self._running_stats_count
        running_std = np.sqrt(running_var)

        normalized_rmse = (rmse - self._running_stats_mean) / (running_std + 1e-8)
        
        param_values = np.array(list(self.current_params.values()), dtype=np.float32)
        return np.append(param_values, normalized_rmse)

    def _get_info(self, rmse):
        """Return additional information."""
        return {"current_rmse": rmse, "current_params": self.current_params}

    def reset(
        self,
        seed=None,
        options=None,
    ):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        self.current_params = self.initial_params.copy()
        self.current_step = 0

        # 運行一次模擬以獲得初始 RMSE
        # initial_i_sim = simulate_eehemt_placeholder(self.current_params, self.vg_values)
        initial_i_sim = self.eehemt_model.functions['ids'].eval(
            temperature = self.temperature,
            voltages = self.sweep_bias,
            **self.current_params,
        )
        self.initial_rmse = self._calculate_rmse(initial_i_sim)
        self.previous_rmse = self.initial_rmse
        
        print(f"Initial Params: {self.current_params}")
        print(f"Initial RMSE: {self.initial_rmse:.4f}%")

        observation = self._get_obs(self.initial_rmse)
        info = self._get_info(self.initial_rmse)
        
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ):
        """Execute one time-step within the environment."""
        self.current_step += 1

        # 1. Update Parameters
        for i, param_name in enumerate(self.tunable_param_names):
            # 將 action（變化量）加到當前參數上
            self.current_params[param_name] += action[i]
            min_val = self.params_config[param_name]['min']
            max_val = self.params_config[param_name]['max']
            self.current_params[param_name] = np.clip(self.current_params[param_name], min_val, max_val)

        # 2. Execute Simulation to get I_sim
        i_sim = self.eehemt_model.functions['ids'].eval(
            temperature = self.temperature,
            voltages = self.sweep_bias,
            **self.current_params,
        )

        # 3. Calculate new RMSE
        current_rmse = self._calculate_rmse(i_sim)

        # 4. Calculate Reward
        # 獎勵 = RMSE 的改善程度，並用初始 RMSE 進行正規化
        reward = (self.previous_rmse - current_rmse) / self.initial_rmse
        
        self.previous_rmse = current_rmse

        # 5. Check Termination and Truncation
        terminated = current_rmse < self.rmse_threshold
        truncated = self.current_step >= self.max_episode_steps
        
        if terminated:
            print(f"Success! RMSE ({current_rmse:.4f}%) has reached the threshold ({self.rmse_threshold:.4f}%).")
        if truncated:
            print("Reached maximum steps.")

        observation = self._get_obs(current_rmse)
        info = self._get_info(current_rmse)

        return observation, reward, terminated, truncated, info
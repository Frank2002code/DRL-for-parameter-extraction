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
        observation_space (gym.spaces.Box): 定義了狀態空間，包括參數值和 RMSPE。
        ...
    """
    metadata = {'render_modes': []}

    def __init__(self, config):
        """
        初始化環境。

        Args:
        """
        super(EEHEMTEnv, self).__init__()

        self.csv_file_path = config.get("csv_file_path", "")
        self.params_config = config.get("tunable_params_config", {})
        self.tunable_param_names = list(self.params_config.keys())
        self.eehemt_model = verilogae.load(config.get("va_file_path", ""))
        self.temperature = 300
        
        measured_data_for_bias = pd.read_csv(self.csv_file_path)
        vgs = measured_data_for_bias['vg'].values
        vds = np.full_like(vgs, 1.5) # 假設 Vds 為 1.5V，請根據您的 csv 檔案調整
        self.sweep_bias = {
            'br_gisi': vgs,
            'br_disi': vds,
            'br_t': vgs,
            'br_esi': vgs
        }

        # 1. 載入真實量測數據 I_meas
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"Measured data file not found:: {self.csv_file_path}")
        measured_data = pd.read_csv(self.csv_file_path)
        self.vg_values = measured_data['vg'].values
        self.i_meas = measured_data['id'].values

        # 2. Define Action Space
        action_deltas = np.array([config['delta'] for config in self.params_config.values()], dtype=np.float32)
        self.action_space = spaces.Box(low=-action_deltas, high=action_deltas, dtype=np.float32)

        # 3. Define Observation Space
        param_mins = [config['min'] for config in self.params_config.values()]
        param_maxs = [config['max'] for config in self.params_config.values()]
        
        low_bounds = np.append(param_mins, -np.inf).astype(np.float32)
        high_bounds = np.append(param_maxs, np.inf).astype(np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        
        # ==================== 主要修改部分 START ====================
        # 4. 初始化內部狀態
        # 根據新邏輯，直接從 Verilog-A 模型中獲取所有參數的預設值作為初始狀態
        self.initial_params = {name: param.default for name, param in self.eehemt_model.modelcard.items()}
        self.current_params = self.initial_params.copy()
        # ===================== 主要修改部分 END =====================

        self.initial_rmspe = -1.0
        self.previous_rmspe = -1.0
        
        self._running_stats_count = 1e-4
        self._running_stats_mean = 0.0
        self._running_stats_M2 = 0.0
        
        self.max_episode_steps = 10000
        self.rmspe_threshold = 0.05
        self.current_step = 0

    def _calculate_rmspe(self, i_sim):
        i_meas_safe = np.where(np.abs(self.i_meas) < 1e-12, 1e-12, self.i_meas)
        rmspe = np.sqrt(np.mean(np.square((self.i_meas - i_sim) / i_meas_safe)))
        return rmspe
    
    def _update_running_stats(self, new_rmspe_value: float):
        self._running_stats_count += 1
        delta = new_rmspe_value - self._running_stats_mean
        self._running_stats_mean += delta / self._running_stats_count
        delta2 = new_rmspe_value - self._running_stats_mean
        self._running_stats_M2 += delta * delta2

    def _get_obs(self, rmspe: float) -> np.ndarray:
        self._update_running_stats(rmspe)
        running_var = self._running_stats_M2 / self._running_stats_count
        running_std = np.sqrt(running_var)
        normalized_rmspe = (rmspe - self._running_stats_mean) / (running_std + 1e-8)
        
        param_values = np.array([self.current_params[name] for name in self.tunable_param_names], dtype=np.float32)
        return np.append(param_values, normalized_rmspe).astype(np.float32)

    def _get_info(self, rmspe):
        return {"current_rmspe": rmspe, "current_params": self.current_params}

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)
        self.current_params = self.initial_params.copy()
        self.current_step = 0

        initial_i_sim = self.eehemt_model.functions['Ids'].eval(
            temperature = self.temperature,
            voltages = self.sweep_bias,
            **self.current_params,
        )
        self.initial_rmspe = self._calculate_rmspe(initial_i_sim)
        self.previous_rmspe = self.initial_rmspe
        
        print(f"Initial Params (tunable part): {{ {', '.join(f'{k}: {v:.4f}' for k, v in {name: self.current_params[name] for name in self.tunable_param_names}.items())} }}")
        print(f"Initial RMSPE: {self.initial_rmspe:.4f}")

        observation = self._get_obs(self.initial_rmspe)
        info = self._get_info(self.initial_rmspe)
        
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ):
        self.current_step += 1

        for i, param_name in enumerate(self.tunable_param_names):
            self.current_params[param_name] += action[i]
            min_val = self.params_config[param_name]['min']
            max_val = self.params_config[param_name]['max']
            self.current_params[param_name] = np.clip(self.current_params[param_name], min_val, max_val)

        i_sim = self.eehemt_model.functions['Ids'].eval(
            temperature = self.temperature,
            voltages = self.sweep_bias,
            **self.current_params,
        )

        current_rmspe = self._calculate_rmspe(i_sim)
        reward = self.previous_rmspe - current_rmspe
        self.previous_rmspe = current_rmspe

        terminated = current_rmspe < self.rmspe_threshold
        truncated = self.current_step >= self.max_episode_steps
        
        if terminated:
            print(f"Success! RMSPE ({current_rmspe:.4f}) has reached the threshold ({self.rmspe_threshold}).")
        if truncated and not terminated:
            print("Reached maximum steps.")

        observation = self._get_obs(current_rmspe)
        info = self._get_info(current_rmspe)

        return observation, reward, terminated, truncated, info
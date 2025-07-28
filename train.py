from env.eehemt_env import EEHEMTEnv
from ray.rllib.algorithms.ppo import PPOConfig

csv_file_path = "/home/u5977862/DRL-on-parameter-extraction/data/S25E02A025WS_25C_GMVG.csv"
tunable_parameters = {
    # === Threshold Voltage & Sub-threshold Parameters ===
    'DVT0': {
        'initial': 0.0, 
        'min': -0.5, 
        'max': 0.5, 
        'delta': 0.01
    },
    'ETA0': {
        'initial': 0.6, 
        'min': 0.1, 
        'max': 2.0, 
        'delta': 0.05
    },
    'DSUB': {
        'initial': 1.06, 
        'min': 0.5, 
        'max': 2.0, 
        'delta': 0.05
    },
    'PHIN': {
        'initial': 0.05, 
        'min': -0.2, 
        'max': 0.2, 
        'delta': 0.01
    },

    # === Mobility Parameters ===
    'U0': {
        'initial': 0.03, 
        'min': 0.01, 
        'max': 0.1, 
        'delta': 0.001
    },
    'UA': {
        'initial': 0.3, 
        'min': -1.0, 
        'max': 1.0, 
        'delta': 0.05
    },

    # === Saturation Effects Parameters ===
    'VSAT': {
        'initial': 85000.0, 
        'min': 50000.0, 
        'max': 150000.0, 
        'delta': 1000.0
    },
    'PDIBL1': {
        'initial': 1.3, 
        'min': 0.0, 
        'max': 5.0, 
        'delta': 0.1
    },
    'PDIBL2': {
        'initial': 0.0002, 
        'min': 0.0, 
        'max': 0.01, 
        'delta': 0.0001
    },

    # === Parasitic Resistance Parameter ===
    'RDSW': {
        'initial': 200.0, 
        'min': 0.0, 
        'max': 500.0, 
        'delta': 10.0
    }
}

# Configure.
config = (
    PPOConfig()
    .environment(
        EEHEMTEnv,
        env_config={
            "csv_file": csv_file_path,
            "tunable_params_config": tunable_parameters,
        },
    )
    .training(
        train_batch_size_per_learner=2000,
        lr=0.0004,
    )
)

# Build the Algorithm.
algo = config.build_algo()

# Train for one iteration, which is 2000 timesteps (1 train batch).
print(algo.train())
checkpoint_dir = algo.save_to_path()
print(f"saved algo to {checkpoint_dir}")


import os

from env.eehemt_env import EEHEMTEnv_Norm_Vtos, tunable_params_config

if __name__ == "__main__":
    config = {
        "csv_file_path": "/home/u5977862/DRL-on-parameter-extraction/data/S25E02A025WS_25C_GMVG.csv",
        "tunable_params_config": tunable_params_config,
        "va_file_path": "/home/u5977862/DRL-on-parameter-extraction/eehemt/eehemt114_2.va",
        "test_modified": True,
    }
    env = EEHEMTEnv_Norm_Vtos(
        config,
    )

    save_path = "./result/test_iv_curve.png"
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    env.plot_iv_curve(
        save_path=save_path,
    )
    print(f"{type(env).__name__} Test Success.")

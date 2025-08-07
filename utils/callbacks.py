# utils/callbacks.py
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from utils.plot import plot_iv_curve



class CustomEvalCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode,
        **kwargs,
    ) -> None:
        last_info = episode.infos[-1]
        if "plot_data" in last_info:
            rmspe = last_info["final_rmspe"]
            plot_data = last_info["plot_data"]

            # episode.custom_data["final_rmspe"] = rmspe
            # episode.custom_data["plot_data"] = plot_data
            print(f"\n=====Final RMSPE: {rmspe:.4f}=====")
            plot_iv_curve(
                plot_data=plot_data,
                plot_initial=True,
                plot_modified=True,
                plot_current=True,
                save_path="result/final_iv_curve.png"
            )

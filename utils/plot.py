import matplotlib.pyplot as plt
import os

def plot_iv_curve(
    plot_data: dict,
    plot_initial: bool = True,
    plot_modified: bool = True,
    plot_current: bool = True,
    save_path: str | None = None,
):
    """
    使用預先計算好的數據來繪製 I-V 曲線圖。
    """
    output_dir = os.path.dirname(save_path) if save_path else "results"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))

    # Plot measured data from the dictionary
    plt.plot(plot_data['vgs'], plot_data['i_meas'], "ko", label="Measured Data (Target)")

    # Plot simulated curves based on options
    if plot_initial:
        plt.plot(plot_data['vgs'], plot_data['i_sim_initial'], "b--", label="Simulated (Initial Params)")

    if plot_modified:
        plt.plot(plot_data['vgs'], plot_data['i_sim_modified'], "g-.", label="Simulated (Modified Initial Params)")

    if plot_current:
        plt.plot(plot_data['vgs'], plot_data['i_sim_current'], "r-", label="Simulated (Current Params)")

    # Style the plot
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
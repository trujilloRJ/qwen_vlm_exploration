import os
import json
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    res_folder = r"C:\jtr\side_projects\LLMs\qwen_vlm_exploration\_results"
    model_name = "Qwen3.5-2B_SFT"
    
    file_path = os.path.join(res_folder, model_name, "log.json")
    with open(file_path, "r") as f:
        logs = json.load(f)

    data = {
        "epoch": [log["epoch"] for log in logs],
        "step": [log["step"] for log in logs],
        "loss": [log["loss"] for log in logs],
        "g_step": [0]
    }
    for log in logs[1:]:
        data["g_step"].append(data["g_step"][-1] + 1)

    mov_avg_window = 30
    loss_ma = np.convolve(data["loss"], np.ones(mov_avg_window)/mov_avg_window, mode='same')

    fig, ax = plt.subplots()
    ax.plot(data["g_step"], data["loss"], label="Training Loss", alpha=0.5)
    ax.plot(data["g_step"], loss_ma, label=f"Moving Average ({mov_avg_window} steps)", color='green')
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss Curve for {model_name}")
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()
    

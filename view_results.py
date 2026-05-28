import json
import os
import matplotlib.pyplot as plt

if __name__=="__main__":
    res_folder = r"C:\jtr\side_projects\LLMs\qwen_vlm_exploration\_results"
    dataset_name = "textvqa"
    models = ["Qwen3.5-2B", "Qwen3.5-2B_SFT", "Qwen3.5-2B_v1", "Qwen3.5-0.8B"]

    metrics = []
    for model_name in models:
        metric = {}
        # get performance metrics
        data_path = os.path.join(res_folder, model_name, f"{dataset_name}_metrics.json")
        with open(data_path, "r") as f:
            data = json.load(f)
        metric["average_vqa_score"] = data["average_score"]

        # get model stats
        data_path = os.path.join(res_folder, model_name, f"{model_name}_stats.json")
        with open(data_path, "r") as f:
            data = json.load(f)
        metric["memory_footprint_gb"] = data["memory_footprint_gb"]

        metrics.append((model_name, metric))

    # Plot the metrics
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    properties = [
        {"color": '#4C72B0', "annotation_x": -250, "annotation_y": -0.01},
        {"color": '#DD8452', "annotation_x": -250, "annotation_y": -0.01},
        {"color": '#55A868', "annotation_x": -100, "annotation_y": 0.02},
        {"color": '#C44E52', "annotation_x": -100, "annotation_y": 0.02},
    ]
    fig, ax = plt.subplots()
    for (model_name, metric), color, prop in zip(metrics, colors, properties):
        x = metric["memory_footprint_gb"] * 1024  # convert GB to MB
        y = metric["average_vqa_score"]
        ax.scatter(x, y, color=color, s=200, zorder=3, label=model_name)
        ax.annotate(f"{y:.2f}", (x+prop["annotation_x"], y+prop["annotation_y"]), textcoords="offset points", xytext=(0, 0), fontsize=12)
    ax.set_xlabel('Memory Footprint (MB)')
    ax.set_ylabel('Average VQA Score')
    ax.set_title('Model Performance vs Memory Footprint on TextVQA')
    ax.set_ylim(0.5, 1)
    ax.legend(loc="lower right")
    # ax.set_xscale('log')
    plt.tight_layout()
    plt.show()
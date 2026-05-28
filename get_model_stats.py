import os
import json
from dotenv import load_dotenv
from utils import get_hqq_quantization_config, get_quantization_config, load_model_and_processor
from quantizations import QUANTIZATIONS

load_dotenv()

if __name__ == "__main__":
    res_folder = r"C:\jtr\side_projects\LLMs\qwen_vlm_exploration\_results"
    models = [
        # {"name": "Qwen/Qwen3.5-0.8B", "quantization": "v0"},  # no quantization
        {"name": "Qwen/Qwen3.5-2B", "quantization": "v0"},  
        {"name": "Qwen/Qwen3.5-2B", "quantization": "v1"},  
        {"name": "Qwen/Qwen3.5-2B", "quantization": "hqq_4bit"},  
    ]

    for model_dict in models:
        model_id = model_dict["name"]
        quantization = model_dict["quantization"]

        quant_params = QUANTIZATIONS[quantization]
        if quantization.startswith("hqq"):
            quant_config = get_hqq_quantization_config(**quant_params) if quant_params else None
        else:
            quant_config = get_quantization_config(**quant_params) if quant_params else None

        model, processor = load_model_and_processor(model_id, quantization_config=quant_config)

        bytes_used = model.get_memory_footprint()
        gigabytes_used = bytes_used / (1024 ** 3)
        
        model_dict["stats"] = {
            "memory_footprint_gb": gigabytes_used,
        }

        model_name = model_id.split("/")[1]
        if quantization != "v0":
            model_name += f"_{quantization}"
        out_path = os.path.join(res_folder, model_name, f"{model_name}_stats.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(model_dict["stats"], f, indent=4)
        print(f"Model: {model_name}, Memory Footprint: {gigabytes_used:.2f} GB")

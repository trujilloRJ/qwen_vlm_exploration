from typing import Literal

from dotenv import load_dotenv

from quantizations import QUANTIZATIONS
from utils import get_quantization_config, load_model_and_processor, format_message, generate_and_decode, load_data_from_disk, load_data_from_hf

load_dotenv()


if __name__ == "__main__":
    model_id = "Qwen/Qwen3.5-0.8B"
    LOCAL = False
    prompt = "Answer briefly based on the image"
    quantization: Literal["v0", "v1"] = "v1" # v0 is no quantization 
    n_samples = 3

    quant_params = QUANTIZATIONS[quantization]
    quant_config = get_quantization_config(**quant_params) if quant_params else None

    model, processor = load_model_and_processor(model_id, quantization_config=quant_config)

    bytes_used = model.get_memory_footprint()
    gigabytes_used = bytes_used / (1024 ** 3)
    print(f"Model Memory Footprint: {gigabytes_used:.2f} GB")

    # if LOCAL:
    #     dataset_path = os.getenv("TEXTVQA_DATA_PATH")
    #     if not dataset_path:
    #         raise ValueError("Please set the TEXTVQA_DATA_PATH environment variable to the path of the TextVQA dataset.")
    #     data = load_data_from_disk(dataset_path, "validation-00000")
    # else:
    #     data = load_data_from_hf()

    # pbar = tqdm.tqdm(data, total=n_samples, desc="Processing samples...")
    # results = []
    # for i, sample in enumerate(pbar):
    #     if n_samples is not None and i >= n_samples:
    #         break
    #     messages = format_message(sample["question"], sample["image"], prompt=prompt)
    #     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     inputs = processor(
    #         text=[text],
    #         images=sample["image"],
    #         padding=True,
    #         return_tensors="pt"
    #     ).to("cuda")

    #     output_text = generate_and_decode(model, processor, inputs)
    #     results.append({
    #         "question_id": sample["question_id"],
    #         "question": sample["question"],
    #         "answers": sample["answers"],
    #         "predicted_answer": output_text,
    #     })

    # # save results to a json file
    # out_path = os.path.join("_results", model_id.split("/")[1], "textvqa.json")
    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # with open(out_path, "w") as f:
    #     json.dump(results, f, indent=4)


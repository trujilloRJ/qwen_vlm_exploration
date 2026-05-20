import os
import glob

from datasets import load_dataset
from dotenv import load_dotenv
import torch
import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import json

load_dotenv()

def format_message(question: str, image, prompt: str = "Answer briefly") -> list:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{prompt}: {question}"}
            ]
        }
    ]
    return messages


def load_model_and_processor(model_id: str):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_data_from_disk(dataset_path: str, files_regex: str, split_name: str = "validation"):
    files = glob.glob(os.path.join(dataset_path, "**", "*.arrow"), recursive=True)
    val_files = [f for f in files if files_regex in f]
    data = load_dataset(
        "arrow",
        data_files={split_name: val_files},
        split=split_name,
    )
    return data


def load_data_from_hf(dataset_name: str = "lmms-lab/textvqa", split: str = "validation"):
    data = load_dataset(dataset_name, split=split, streaming=True)
    return data


def generate_and_decode(model, processor, inputs):
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100, pad_token_id=processor.tokenizer.eos_token_id)
        # Trim out the prompt tokens to decode only the predictions
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
    return output_text


if __name__ == "__main__":
    model_id = "Qwen/Qwen3.5-0.8B"
    LOCAL = False
    prompt = "Answer briefly based on the image"
    n_samples = 3

    model, processor = load_model_and_processor(model_id)

    if LOCAL:
        dataset_path = os.getenv("TEXTVQA_DATA_PATH")
        if not dataset_path:
            raise ValueError("Please set the TEXTVQA_DATA_PATH environment variable to the path of the TextVQA dataset.")
        data = load_data_from_disk(dataset_path, "validation-00000")
    else:
        data = load_data_from_hf()

    pbar = tqdm.tqdm(data, total=n_samples, desc="Processing samples...")
    results = []
    for i, sample in enumerate(pbar):
        if n_samples is not None and i >= n_samples:
            break
        messages = format_message(sample["question"], sample["image"], prompt=prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=sample["image"],
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        output_text = generate_and_decode(model, processor, inputs)
        results.append({
            "question_id": sample["question_id"],
            "question": sample["question"],
            "answers": sample["answers"],
            "predicted_answer": output_text,
        })

    # save results to a json file
    out_path = os.path.join("_results", model_id.split("/")[1], "textvqa.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)


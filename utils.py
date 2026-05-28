import glob
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, HqqConfig
from datasets import load_dataset

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


def get_quantization_config(modules_to_skip: list[str] | None = None, **kwargs) -> BitsAndBytesConfig:
    if modules_to_skip is None:
        modules_to_skip = []
    else:
        # lm_head MUST be always here, since llm_int8_skip_modules clean the default, see open PR issue
        # https://github.com/huggingface/transformers/issues/45674
        # e.g. modules_to_skip = ["lm_head", "vision_tower", "multi_modal_projector"]
        modules_to_skip.append("lm_head")
    
    default_kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": False,
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_quant_type": "nf4",
    }
    default_kwargs.update(kwargs)

    quantization_config = BitsAndBytesConfig(
        llm_int8_skip_modules=modules_to_skip,
        **default_kwargs 
    )

    return quantization_config


def get_hqq_quantization_config(modules_to_skip: list[str] | None = None, nbits: int = 4) -> HqqConfig:
    if modules_to_skip is None:
        modules_to_skip = ["lm_head"]
    else:
        # lm_head MUST be always here, since llm_int8_skip_modules clean the default, see open PR issue
        modules_to_skip.append("lm_head")
    
    quant_config = HqqConfig(
        nbits=nbits, 
        group_size=64, 
        axis=1, 
        skip_modules=modules_to_skip
    ) 
    return quant_config


def _apply_hqq_to_model(model, hqq_config):
    """Apply HQQ quantization directly using the hqq library.

    Bypasses transformers' HqqConfig integration (not yet available in
    newer transformers) and replaces nn.Linear layers with HQQLinear,
    skipping any module whose full name contains an entry from skip_modules.
    """
    import torch.nn as nn
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
    from hqq.models.base import find_parent

    skip_modules = hqq_config.skip_modules or ["lm_head"]
    wparams = hqq_config.quant_config['weight_quant_params']
    quant_config = BaseQuantizeConfig(
        nbits=wparams['nbits'], 
        group_size=wparams['group_size']
    )

    # HQQLinear requires module.name to be set
    for name, module in model.named_modules():
        module.name = name

    # Collect layers to replace before modifying the model
    to_replace = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not any(skip in name for skip in skip_modules)
    }

    for name, module in to_replace.items():
        device = str(next(module.parameters()).device)
        hqq_linear = HQQLinear(
            module, quant_config, compute_dtype=torch.bfloat16, device=device
        )
        setattr(find_parent(model, name), name.split(".")[-1], hqq_linear)

    model.hqq_quantized = True


def load_model_and_processor(model_id: str, quantization_config=None):
    if quantization_config is None:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
        )
    elif isinstance(quantization_config, HqqConfig):
        # transformers' HQQ quantizer raises NotImplementedError in newer versions;
        # load in bfloat16 and apply HQQ manually via the hqq library instead.
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        _apply_hqq_to_model(model, quantization_config)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
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
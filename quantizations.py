QUANTIZATIONS = {
    "v0": None, # No quantization
    "v1": {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": False,
        "bnb_4bit_quant_type": "nf4",
        "modules_to_skip": ["vision_tower", "multi_modal_projector"]
    },
    "hqq_4bit": {
        "nbits": 4,
        "modules_to_skip": ["vision_tower", "multi_modal_projector"]
    }
}
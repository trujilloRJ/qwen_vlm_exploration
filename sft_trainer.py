import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from utils import load_data_from_hf, load_model_and_processor

load_dotenv()

def textvqa_collate_fn(batch_list, processor):
    """
    Takes a list of dataset dictionaries and returns a tensor batch.
    Images are passed through the processor alongside text so the model
    sees the visual tokens. Prompt + padding tokens are masked with -100
    so they are ignored during loss calculation.
    """
    questions = [item["question"] for item in batch_list]
    images = [item["image"] for item in batch_list]

    # TextVQA answers are usually a list of strings for each image
    # We find the most common answer (majority vote)
    best_answers = []
    for item in batch_list:
        answers = item["answers"]
        if isinstance(answers, list) and len(answers) > 0:
            best_answers.append(max(set(answers), key=answers.count))
        else:
            best_answers.append("")  # Fallback just in case

    # Build full conversation messages (user + assistant)
    messages_full = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"Answer briefly based on the image: {q}"}
                ]
            },
            {"role": "assistant", "content": ans}
        ]
        for q, img, ans in zip(questions, images, best_answers)
    ]

    texts_full = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in messages_full
    ]

    # Tokenize full sequences with left-padding
    processor.tokenizer.padding_side = "left"
    encoded = processor(
        text=texts_full,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]

    # Build labels: mask padding + prompt tokens, keep only answer tokens.
    # We locate the assistant turn boundary by finding the last occurrence of
    # the <|im_start|> token (which precedes "assistant\n") in each sequence,
    # avoiding a second per-sample processor call.
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    labels = input_ids.clone()
    for i in range(len(batch_list)):
        # Find the last <|im_start|> — that is the assistant turn header
        positions = (input_ids[i] == im_start_id).nonzero(as_tuple=True)[0]
        # Mask everything up to and including that token (header + "assistant\n")
        # +2 accounts for the "assistant" and newline tokens that follow
        assistant_start = positions[-1].item() + 3  # <|im_start|> + "assistant" + "\n"
        labels[i, :assistant_start] = -100

    # Return all processor outputs so pixel_values etc. reach the model
    return {**encoded, "labels": labels}

if __name__ == "__main__":
    device = "cuda"
    model_id = "Qwen/Qwen3.5-2B"
    lr = 2e-5
    n_epochs = 3
    batch_size = 2
    n_train_samples = 500
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["up_proj", "down_proj", "gate_proj"], # Trying only on the MLP layers
        bias="none",
        task_type="CAUSAL_LM"
    )

    # DEBUG
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sample_text = "nokia<|im_end|>"
    tokens = tokenizer(sample_text, add_special_tokens=False).input_ids
    print(f"Tokens for '{sample_text}': {tokens}") #=> [77, 26658, 248046]
    # =================

    model, processor = load_model_and_processor(model_id)

    # Set up LoRA configuration and apply it to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print the number of trainable parameters
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    # Preparing data
    train_loader = load_data_from_hf(split="train") # in streaming
    train_loader = DataLoader(
        train_loader, 
        batch_size=batch_size, 
        collate_fn=lambda x: textvqa_collate_fn(x, processor)
    )

    # Training loop
    log = []
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        pbar = tqdm(train_loader, desc="Batches", total=n_train_samples // batch_size)
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            log.append({
                'epoch': epoch,
                'step': i, 
                'loss': loss.item()
            })

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            if i * batch_size >= n_train_samples:
                break

    # Saving model 
    model.save_pretrained(f"{model_id.split('/')[1]}_lora_finetuned")
            

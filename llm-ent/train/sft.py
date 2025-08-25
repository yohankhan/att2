import json, yaml, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

def load_sft_dataset(path):
    # JSONL with {"prompt": "...", "response": "..."}
    return load_dataset("json", data_files=path, split="train")

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    bnb = None
    if cfg.get("use_qlora", False):
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb, device_map="auto")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tok.padding_side = "right"; tok.pad_token = tok.eos_token

    ds = load_sft_dataset(cfg["dataset_jsonl"]).map(
        lambda ex: {"text": f"<s>[INST] {ex['prompt']} [/INST]\n{ex['response']}</s>"}
    )

    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        max_seq_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=cfg["lr"],
        bf16=cfg["bf16"],
        logging_steps=10,
        save_steps=1000
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds, args=sft_cfg, dataset_text_field="text"
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    import sys; main(sys.argv[1])

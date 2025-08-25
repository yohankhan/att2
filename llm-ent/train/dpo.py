import yaml, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

def load_dpo(path):
    # JSONL with {"prompt": "...", "chosen": "...", "rejected": "..."}
    return load_dataset("json", data_files=path, split="train")

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto", torch_dtype="bfloat16")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tok.pad_token = tok.eos_token

    ds = load_dpo(cfg["dataset_jsonl"]).map(lambda ex: {
        "prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]
    })

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # will clone by default
        args=DPOConfig(
            output_dir=cfg["output_dir"],
            beta=cfg.get("beta", 0.1),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=cfg["lr"],
            bf16=cfg["bf16"],
            max_steps=cfg["max_steps"],
            save_steps=1000
        ),
        tokenizer=tok,
        train_dataset=ds
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    import sys; main(sys.argv[1])

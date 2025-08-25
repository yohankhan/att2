import os, yaml, math, json
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model.build_model import build_from_yaml, init_for_training
from accelerate import Accelerator

class PackedParquetDataset(Dataset):
    def __init__(self, parquet_path):
        self.tbl = pq.read_table(parquet_path)
        self.ids = self.tbl["input_ids"].to_pylist()
        self.attn = self.tbl["attention_mask"].to_pylist()
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        x = torch.tensor(self.ids[i], dtype=torch.long)
        attn = torch.tensor(self.attn[i], dtype=torch.long)
        return {"input_ids": x, "attention_mask": attn, "labels": x.clone()}

def main(cfg_path: str):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    accelerator = Accelerator(mixed_precision="bf16" if cfg["bf16"] else "no")
    model = build_from_yaml(cfg["model_config"])
    model = init_for_training(model)

    ds = PackedParquetDataset(cfg["dataset_parquet"])
    dl = DataLoader(ds, batch_size=cfg["batch_size_per_device"], shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(ds) / (cfg["batch_size_per_device"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg["warmup_steps"], total_steps)

    model, optimizer, dl, scheduler = accelerator.prepare(model, optimizer, dl, scheduler)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    step = 0
    model.train()
    for epoch in range(cfg["epochs"]):
        for batch in dl:
            outputs = model(**batch)
            loss = outputs.loss / cfg["grad_accum"]
            accelerator.backward(loss)
            if (step + 1) % cfg["grad_accum"] == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if step % cfg["save_every_steps"] == 0 and accelerator.is_main_process:
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(os.path.join(cfg["output_dir"], f"step_{step}"), safe_serialization=True)
            step += 1
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(os.path.join(cfg["output_dir"], "latest"), safe_serialization=True)

if __name__ == "__main__":
    import sys; main(sys.argv[1])

from transformers import AutoConfig, AutoModelForCausalLM
import yaml, torch

def build_from_yaml(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Use a Llama-like config if desired; AutoConfig works for custom dicts too.
    config = AutoConfig.from_dict(cfg)
    model = AutoModelForCausalLM.from_config(config)
    return model

def init_for_training(model):
    # Enable gradient checkpointing & compile (PyTorch 2.x) for speed
    model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model

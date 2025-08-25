from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", required=True)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
out = m.generate(**tok(args.prompt, return_tensors="pt").to(m.device),
                 max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=True))

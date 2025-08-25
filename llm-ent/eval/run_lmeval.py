"""
Thin wrapper around lm-eval-harness. Requires model path or HF ID.
Example:
  python eval/run_lmeval.py --model checkpoints/dpo --tasks hellaswag,arc_easy
"""
import argparse, subprocess, sys, os
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--tasks", default="hellaswag,arc_easy,piqa,winogrande,boolq")
parser.add_argument("--batch_size", default="auto")
args = parser.parse_args()

cmd = [
    "lm-eval",
    "--model", "hf-causal",
    "--model_args", f"pretrained={args.model},dtype=bfloat16",
    "--tasks", args.tasks,
    "--batch_size", args.batch_size,
    "--output_path", "eval/results.json"
]
print("Running:", " ".join(cmd))
sys.exit(subprocess.call(cmd))

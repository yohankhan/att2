#!/usr/bin/env bash
set -euo pipefail
accelerate launch --config_file configs/deepspeed/ds_z3_offload.json \
  train/dpo.py configs/training/dpo.yaml

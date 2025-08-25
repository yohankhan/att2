#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
accelerate launch --config_file configs/deepspeed/ds_z3_offload.json \
  train/pretrain.py configs/training/pretrain.yaml

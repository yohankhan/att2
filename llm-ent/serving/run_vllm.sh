#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${1:-checkpoints/dpo}
API_KEY=${API_KEY:-"dev-key"}   
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --api-key "$API_KEY" \
  --enable-chunked-prefill \
  --enforce-eager \
  --host 0.0.0.0 --port 8000

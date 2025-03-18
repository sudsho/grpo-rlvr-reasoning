#!/usr/bin/env bash
# Start vLLM as the rollout server on port 8000.
# Trainer's VLLMClient talks to this.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Math-1.5B}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

exec vllm serve "$MODEL" \
    --served-model-name policy \
    --host 0.0.0.0 --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --dtype bfloat16

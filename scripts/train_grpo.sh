#!/usr/bin/env bash
# GRPO training. Requires the vLLM rollout server to be running.
#   Terminal 1: bash scripts/serve_rollout_vllm.sh
#   Terminal 2: bash scripts/train_grpo.sh configs/grpo_math.yaml
set -euo pipefail

CFG="${1:-configs/grpo_math.yaml}"
NUM_GPUS="${NUM_GPUS:-3}"     # 1 GPU is held by vLLM

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    -m src.training.entry --config "$CFG"

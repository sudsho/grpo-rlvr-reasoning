#!/usr/bin/env bash
# SFT warmup on a small curated CoT dataset so the base model reliably
# emits <think>...</think> plus a boxed answer before GRPO takes over.
set -euo pipefail

CFG="${1:-configs/sft_warmup.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    -m src.training.sft_warmup --config "$CFG"

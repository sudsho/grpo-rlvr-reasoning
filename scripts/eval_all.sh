#!/usr/bin/env bash
# Run before+after eval and render the results table.
#
#   BASE_CKPT=src/models/checkpoints/sft_warmup/final \
#   AFTER_CKPT=src/models/checkpoints/grpo_math/final \
#   bash scripts/eval_all.sh
set -euo pipefail

BASE_CKPT="${BASE_CKPT:?set BASE_CKPT}"
AFTER_CKPT="${AFTER_CKPT:?set AFTER_CKPT}"
OUT_DIR="${OUT_DIR:-benchmarks/runs/$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT_DIR/base" "$OUT_DIR/after"

python -m src.eval.run_eval --engine hf --model "$BASE_CKPT"  --out "$OUT_DIR/base"
python -m src.eval.run_eval --engine hf --model "$AFTER_CKPT" --out "$OUT_DIR/after"

python -m src.eval.report_gen \
    --base "$OUT_DIR/base/summary.json" \
    --after "$OUT_DIR/after/summary.json" \
    --out-md   benchmarks/results.md \
    --out-json benchmarks/results.json \
    --model "$AFTER_CKPT" \
    --base-ckpt "$BASE_CKPT" \
    --after-ckpt "$AFTER_CKPT"

echo "wrote benchmarks/results.md and benchmarks/results.json"

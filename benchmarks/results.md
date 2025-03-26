# Eval report (baseline)

Model: `Qwen/Qwen2.5-Math-1.5B` (SFT warmup checkpoint)
Base ckpt: `src/models/checkpoints/sft_warmup/final`
After ckpt: pending overnight GRPO run

Preliminary numbers, will refresh once the GRPO run wraps.

| Benchmark | Baseline pass@1 |
|---|---:|
| gsm8k | 0.612 |
| math | 0.184 |
| mbpp | 0.421 |
| humaneval | 0.396 |

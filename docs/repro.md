# Reproduce

## Prereqs

- Linux (tested Ubuntu 22.04). Windows works for tests but not training.
- CUDA 12.4+ with 4x H100 80GB recommended for the full run.
- Python 3.11 or 3.12.
- (Optional but recommended) `firejail` for a real sandbox.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Smoke test (no GPU needed)

```bash
pytest -q                       # unit tests
```

## SFT warmup

```bash
NUM_GPUS=4 bash scripts/train_sft_warmup.sh configs/sft_warmup.yaml
```

Expected: ~1 hour on 4x H100. Produces
`src/models/checkpoints/sft_warmup/final`.

## GRPO training

Two terminals.

Terminal 1 (rollout server, 1 GPU):

```bash
MODEL=src/models/checkpoints/sft_warmup/final \
GPU_MEM_UTIL=0.85 \
bash scripts/serve_rollout_vllm.sh
```

Terminal 2 (trainer, 3 GPUs):

```bash
NUM_GPUS=3 bash scripts/train_grpo.sh configs/grpo_math.yaml
```

Expected: ~36 hours for 2000 steps on 3x H100. Wandb streams
reward/mean, reward/std, opt/kl, plus the reward histogram.

## Eval

```bash
BASE_CKPT=src/models/checkpoints/sft_warmup/final \
AFTER_CKPT=src/models/checkpoints/grpo_math/final \
bash scripts/eval_all.sh
```

Writes `benchmarks/results.md` and `benchmarks/results.json`.

## Total compute footprint

- SFT warmup: ~1 hr on 4x H100
- GRPO combined run: ~60 hrs on 4x H100 (3 GPUs training + 1 GPU vLLM)
- Eval (all four benches, base + after): ~4 hrs on 1x H100

Rough total: 3 days on a single 4x H100 node.

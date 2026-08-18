# grpo-rlvr-reasoning

GRPO + RLVR post-training on math and code, riding the wave after
DeepSeek R1.

Verifiable rewards from a sandboxed Python executor for code and an
extract-then-compare verifier for math. Rollouts served by vLLM. Base
model is Qwen2.5-Math-1.5B.

## Quick start (tiny-CPU smoke, no GPU/download)

The headline numbers below need a GPU (see "Compute footprint"). But the
machinery (both verifiers, the rule-based/format/composite rewards, and one
GRPO policy update) runs on CPU in seconds with no model download. This is
what a clone-and-run person should try first:

```bash
pip install -e ".[dev]"     # torch, sympy, pytest, ruff (all CPU wheels)
python scripts/smoke_cpu.py # or: make smoke
pytest -q                   # 35 passed, 2 skipped
```

`scripts/smoke_cpu.py` runs the real verifiers + rewards over the bundled
sample rollouts in `data/samples/`, then takes one GRPO-style step on a
from-scratch 2-layer toy causal LM with a mocked rollout (no Qwen, no vLLM).
Real output:

```
========================================================================
1. MATH VERIFIER + REWARDS (bundled GSM8K + MATH rollouts)
========================================================================

[gsm8k_samples.jsonl]  (10 rollouts)
   #        gold  verdict   reward  correct  format  reason
   0          72     PASS    1.100      1.0     1.0  string match
   ...
   9         460     PASS    1.100      1.0     1.0  string match

[math_samples.jsonl]  (5 rollouts)
   3   8\sqrt{2}     PASS    1.100      1.0     1.0  string match

  math verifier: 15/15 rollouts verified correct

========================================================================
2. CODE VERIFIER (sandboxed executor over bundled MBPP rollouts)
========================================================================

[mbpp_samples.jsonl]  (5 rollouts, run in sandbox)
   #  verdict  reward     ms  expected  reason
   0     PASS     1.0    140      True  ok
   1     PASS     1.0    125      True  ok
   2     FAIL     0.0    125     False  exit=1
   3     PASS     1.0    109      True  ok
   4     PASS     1.0    125      True  ok

  code verifier: 4/5 rollouts pass their unit tests; 5/5 match the recorded verdict

========================================================================
3. TINY GRPO STEP (toy causal LM, mocked rollout, CPU)
========================================================================

  toy model: from-scratch 2-layer causal LM (byte tokenizer, d_model=32)
  prompts: 3   group size: 4   samples: 12

  per-group rewards -> group-relative advantages:
    group 0: rewards=[1.100, 1.050, 0.100, 0.000]  adv=[+1.046, +0.948, -0.900, -1.094]

  reward mean : 0.5625  (std 0.5140)
  GRPO loss   : 29.553152 -> 28.609741 (after one Adam step)
  mean |dparam|: 6.801e-04  (params moved => backward + step executed)
```

The one MBPP `FAIL` is a genuinely wrong candidate in the sample set; the
smoke reproduces the recorded verdict (5/5 match). The GRPO loss drops after
one Adam step and the parameters move, so the reward -> advantage -> loss ->
optimizer plumbing is proven end to end on CPU.

**The headline pass@1 gains below are NOT produced by this smoke.** They need
Qwen2.5-Math-1.5B trained with GRPO on a GPU node (a single 4x H100 80GB box,
~3 days). The CPU smoke only proves the loop wiring.

## Results

| Benchmark | Baseline pass@1 | After GRPO | Delta |
|---|---:|---:|---:|
| GSM8K     | 0.612 | 0.734 | **+12** |
| MATH      | 0.184 | 0.321 | **+14** |
| MBPP      | 0.421 | 0.510 | **+9**  |
| HumanEval | 0.396 | 0.470 | **+7**  |

See `benchmarks/results.md` for the full table (per-subject MATH
breakdown too).

## Problem

DeepSeek R1 (Jan 2025) showed that RL from a rule-based verifier is a
real technique for pushing a small reasoning model past its SFT ceiling
on math and code, without a learned reward model. This repo reproduces
a compact GRPO trainer with verifiable rewards on Qwen2.5-Math-1.5B,
targeting GSM8K, MATH, MBPP, HumanEval.

The point is not to chase R1's numbers on a 1.5B base. The point is to
have a readable, working end-to-end loop: prompt to rollout to verifier
to reward to policy update, with the sharp edges (reward hacking,
format drift, length inflation, verifier false positives on code
timeouts) actually addressed. See `docs/reward_hacking_notes.md` for
the running log of what went wrong and how we fixed each.

## Method (short)

Two-stage post-training:

1. **SFT warmup** on a small filtered CoT dataset (~2k traces) to get
   the base model to reliably emit `<think>...</think>` plus a final
   boxed answer / python code block.
2. **GRPO** with group size 8 rollouts per prompt. Composite reward:
   - correctness (verifier pass, 0 or 1)
   - format (well-formed think + answer tag)
   - length penalty (soft ramp, discourages padding)

Rollouts are served by vLLM (async, batched prompts). The policy is
trained with TRL's `GRPOTrainer` wrapped with our custom reward stack.

See `docs/method.md` for the full write-up including the GRPO objective.

## Verifiers

- **Math**: extract-then-compare. Pulls the *last* boxed answer,
  normalizes LaTeX (`\frac`, `\sqrt`, `\pi`, percents, fractions), then
  compares by string, sympy equality, and finally numeric closeness.
- **Code**: runs the candidate against the benchmark's unit tests
  inside a sandboxed subprocess. Backend preference:
  `firejail` > `docker` > plain `subprocess` with rlimit. No network.
  Wall-clock timeout of 5s.

## Compute footprint

Single 4x H100 80GB node, ~3 days end-to-end.

- SFT warmup ~1 hr
- GRPO combined run ~60 hrs (3 GPUs train + 1 GPU vLLM)
- Full eval before + after ~4 hrs

## Repro

See `docs/repro.md`. Short version:

```bash
pip install -r requirements.txt && pip install -e ".[dev]"
NUM_GPUS=4 bash scripts/train_sft_warmup.sh
# terminal 1
bash scripts/serve_rollout_vllm.sh
# terminal 2
NUM_GPUS=3 bash scripts/train_grpo.sh configs/grpo_math.yaml
# after training
bash scripts/eval_all.sh
```

## Training curve

![training curve placeholder](docs/training_curve.png)

wandb: reward/mean climbs from 0.42 (SFT warmup baseline) to 0.72 by
step 800 and plateaus, with kl staying below 0.15. See notebooks/ for
ablations.

## Repo layout

```
src/
  data/         gsm8k, math_bench, mbpp, humaneval loaders + prompts
  verifiers/    latex_norm, math_verifier, sandbox, code_verifier
  rewards/      rule_based, format, composite
  training/     config, collator, rollout, vllm_client, hf_client,
                grpo_trainer, sft_warmup, train_loop, entry
  eval/         run_eval, compare, report_gen
configs/        grpo_math.yaml, grpo_code.yaml, grpo_combined.yaml,
                sft_warmup.yaml
scripts/        train_sft_warmup.sh, train_grpo.sh, eval_all.sh,
                serve_rollout_vllm.sh
tests/          test_math_verifier, test_code_verifier,
                test_sandbox_malicious, test_rewards, test_collator,
                test_grpo_step, test_eval
docs/           method.md, reward_hacking_notes.md, repro.md
notebooks/      analyze_rollouts.ipynb, reward_shaping_ablation.ipynb
data/samples/   10 gsm8k + 5 mbpp + 5 math sampled rollouts
benchmarks/     results.md, results.json
```

## License

MIT.

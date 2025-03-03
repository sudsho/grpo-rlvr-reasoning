# grpo-rlvr-reasoning

GRPO + RLVR post-training on math and code.

## Problem

DeepSeek R1 (Jan 2025) showed that RL from a rule-based verifier is a real
technique for pushing a small reasoning model past its SFT ceiling on math and
code, without a learned reward model. This repo reproduces a compact GRPO
trainer with verifiable rewards on Qwen2.5-Math-1.5B, targeting GSM8K, MATH,
MBPP, HumanEval.

The point is not to chase R1's numbers on a 1.5B base. The point is to have a
readable, working end-to-end loop: prompt to rollout to verifier to reward to
policy update, with the sharp edges (reward hacking, format drift, length
inflation, verifier false positives on code timeouts) actually addressed.

## Method

Two-stage post-training:

1. SFT warmup on a small filtered CoT dataset (a few thousand traces from
   OpenR1 style dumps) to get the base model to reliably emit `<think>...</think>`
   plus a final boxed answer.
2. GRPO with a group size of 8 rollouts per prompt. Reward is a weighted
   combination:
   - correctness (verifier pass, 0 or 1)
   - format (well-formed think + answer tags)
   - a small length penalty against reward hacking via long padding

Rollouts are served by vLLM (async, batched prompts). The policy is trained
with TRL's `GRPOTrainer` wrapped with our custom reward stack.

## Verifiers

- Math: extract-then-compare. Pull the boxed answer, normalize LaTeX
  (\frac, \sqrt, \pi, fractions), and check equality symbolically with sympy.
- Code: run the candidate against the benchmark's unit tests inside a
  sandboxed Python subprocess with rlimit for CPU and memory and a hard
  wall-clock timeout. Stdio is captured. No network.

## Results

TODO after training run finishes.

## Repro

TODO. Will land in `docs/repro.md`.

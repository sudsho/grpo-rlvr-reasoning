# grpo-rlvr-reasoning

GRPO + RLVR post-training on math and code, riding the wave after DeepSeek R1.

Reproducing a compact GRPO trainer with rule-based verifiable rewards. Base
model is Qwen2.5-Math-1.5B. Rollouts are served by vLLM. Rewards come from a
sandboxed Python executor for code and an extract-then-compare verifier for
math.

status: work in progress. more soon.

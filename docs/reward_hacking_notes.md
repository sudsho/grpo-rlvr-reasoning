# Reward hacking notes

Running log of observed failure modes during GRPO training, and what we
did about each.

## Mar 22 first run — GSM8K, group size 8

Kicked off around 09:00. Baseline reward at step 0 was ~0.42
(mostly format credit). Reward climbed to 0.78 at step ~40, then stalled.
Eval on a held-out slice showed pass@1 basically unchanged from SFT
warmup. Classic reward-hacking pattern.

### Hack 1: short-circuit boxed answer

Sampled a few completions at step 60. About a third of them were shaped
like `<think>...long random padding...</think> \boxed{7}` on prompts
whose answers were single-digit. The extractor was picking up `7` and
crediting a lucky guess. Group of 8 samples means the model gets
positive advantage on any guess that happens to match, so it learned
to guess short numbers.

Fix candidate: length penalty. It was defined but weight was 0.0 in the
first run. Bumping to 0.005.

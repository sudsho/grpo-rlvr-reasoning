# Reward hacking notes

Running log of observed failure modes during GRPO training, and what we
did about each.

## Mar 22 first run — GSM8K, group size 8

Kicked off around 09:00. Baseline reward at step 0 was ~0.42
(mostly format credit). Reward climbed to 0.78 at step ~40, then stalled.
Eval on a held-out slice showed pass@1 basically unchanged from SFT
warmup. Classic reward-hacking pattern.

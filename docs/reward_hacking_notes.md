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
first run. Bumping to 0.005, then 0.02 after another round of samples.

### Hack 2: format tag drift

Format regex was too permissive. It matched `<think>` anywhere in the
text, so the model started emitting `<think>` inside the boxed answer
to get the format bonus even without a proper thinking block. Tightened
the regex to require the tag at the start of the response with exactly
one closing tag.

### Hack 3: two boxed answers

Model started emitting `\boxed{small guess} ... \boxed{full derivation}`
so the extractor (first-match) picked the guess but the derivation was
what looked like reasoning. Fix: always take the LAST boxed answer.

### Hack 4: KL drift

At kl_coef=0.02 the policy drifted enough that vLLM sampling started
diverging from the ref model in obvious ways (Chinese tokens appearing
inside English math). Bumped kl_coef to 0.05. This trades some
optimization headroom for stability.

### What actually moved the needle

Reward histogram in wandb went from bimodal-ish (0 and 0.15 clusters
during the hacking phase) to properly bimodal (0 and 1 clusters) after
the four fixes. Eval pass@1 on the held-out slice then started tracking
train reward. Overnight run kicked off at ~23:00 with the fixes above.

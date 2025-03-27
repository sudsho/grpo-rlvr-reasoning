# Method: GRPO with rule-based verifiable rewards

## Setting

Base model: Qwen2.5-Math-1.5B (chosen because it fits on a single H100 for
training with room for a ref-copy, and has a reasonable math prior). We
target four evaluation benchmarks:

- GSM8K (grade-school word problems)
- MATH (competition-level, subject-tagged)
- MBPP (basic python problems with unit tests)
- HumanEval (function-completion with unit tests)

## Stage 1: SFT warmup

A short SFT pass on ~2k CoT traces (openr1-math + a small MBPP filter) so
the model reliably emits `<think>...</think>` plus a boxed answer or a
python code fence. This is not to boost benchmark numbers; it's to give
GRPO a policy that already gets partial reward from the format term.
Without warmup, the format term is 0 for the first few hundred rollouts
and the correctness signal is too sparse to bootstrap.

## Stage 2: GRPO

For each prompt `q` we sample `G = 8` responses under the current policy
`pi_theta_old`. We compute a reward for each with our composite reward:

```
r_i = w_c * correctness(o_i) + w_f * format(o_i) + w_l * length_penalty(o_i)
```

with defaults `w_c=1.0`, `w_f=0.1`, `w_l=0.005`. Rewards are turned into
group-relative advantages:

```
A_i = (r_i - mean(r)) / (std(r) + eps)
```

The GRPO objective (following DeepSeek-Math notation) is:

```
J = E[ 1/G sum_i min( ratio_i A_i, clip(ratio_i, 1-eps, 1+eps) A_i ) ]
    - beta * KL(pi_theta || pi_ref)
```

where `ratio_i = pi_theta(o_i|q) / pi_theta_old(o_i|q)` and `pi_ref` is a
frozen copy of the SFT-warmup model. We use `beta = 0.05` (bumped from
0.02 after observing drift, see `reward_hacking_notes.md`).

## Verifiers

### Math

Extract-then-compare. Steps:

1. Pull the *last* `\boxed{...}` from the response (models sometimes emit
   two, and the last one is where they've committed to an answer).
2. Normalize both prediction and gold: strip LaTeX wrappers, expand
   `\frac`, convert `^` to `**`, drop degree symbols, collapse whitespace.
3. Compare in three phases: string equality, sympy `simplify(a-b)==0`,
   numeric closeness with tolerance `1e-6`.

### Code

Sandboxed Python subprocess:

- Preferred backend: `firejail` with `--net=none` and rlimit-as.
- Fallback backend: plain `subprocess` with `resource.setrlimit` and a
  wall-clock timeout (5s per test).
- Windows fallback: no rlimit, timeout only. Enough for CI.

For MBPP we compose `setup + candidate + test_list` into one script and
run it. For HumanEval we compose `candidate + test + check(entry_point)`.

## Rollout

vLLM's OpenAI-compatible HTTP endpoint. The trainer talks to it over
`aiohttp` with a semaphore for concurrency (default 32 in-flight
requests). Between epochs we swap the served weights by writing new
safetensors to a shared path and issuing a lightweight reload.

For dev boxes and CI we ship an HF-generate fallback with the same
interface.

## Reference points

- DeepSeek-Math (Shao et al., 2024) introduced GRPO in a math setting.
- DeepSeek R1 (Jan 2025) demonstrated that RL from rule-based verifiable
  rewards, without a learned reward model, is sufficient to elicit strong
  reasoning behaviors from a base LM.
- OpenAI o1 and o3 use closed post-training pipelines; the public
  reproduction wave (OpenR1, TinyZero, SimpleRL) demonstrates that
  compact GRPO trainers can be built with off-the-shelf pieces.

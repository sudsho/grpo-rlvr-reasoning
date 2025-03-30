# Eval report

Model: `Qwen/Qwen2.5-Math-1.5B`
Base ckpt: `src/models/checkpoints/sft_warmup/final`
After ckpt: `src/models/checkpoints/grpo_combined/final`
Trained: 3000 GRPO steps, group_size=8, ~60 hrs on 4x H100.

| Benchmark | Baseline pass@1 | After GRPO pass@1 | Delta |
|---|---:|---:|---:|
| GSM8K     | 0.612 | 0.734 | +0.122 |
| MATH      | 0.184 | 0.321 | +0.137 |
| MBPP      | 0.421 | 0.510 | +0.089 |
| HumanEval | 0.396 | 0.470 | +0.074 |

Headline: +14 pt on MATH, +9 pt on MBPP, GSM8K jumps 12 pt from a
warmed-up SFT baseline. HumanEval moves less (base was already fairly
saturated for a 1.5B code model).

## Baseline vs after by MATH subject (approximate)

| Subject | Baseline | After |
|---|---:|---:|
| Algebra          | 0.24 | 0.41 |
| Counting/Prob    | 0.17 | 0.29 |
| Geometry         | 0.11 | 0.19 |
| Intermediate Alg | 0.15 | 0.28 |
| Num Theory       | 0.21 | 0.36 |
| Prealgebra       | 0.35 | 0.53 |
| Precalculus      | 0.09 | 0.16 |

Precalculus and Geometry lag; they're the two subjects where the
verifier's LaTeX comparator is weakest (open questions about equivalent
forms of an expression, e.g. `\sin^2 + \cos^2` collapsing to 1).

## Compute footprint

- SFT warmup: ~1 hour on 4x H100
- GRPO training: ~60 hours on 4x H100 (3 train + 1 vLLM)
- Eval (all 4 benches, base + after): ~4 hours on 1x H100

Rough total: 3 days on a single 4x H100 node.

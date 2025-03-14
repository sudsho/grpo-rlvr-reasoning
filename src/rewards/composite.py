"""Weighted combo reward.

r_total = w_c * correctness + w_f * format + w_l * length_penalty(response)

Weights come from config. Defaults:
- correctness: 1.0 (the main signal)
- format:      0.1 (small; too high and the model gets rewarded for tags)
- length:      0.005 (very small; too high and it collapses to one-token
                     answers that happen to match)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class RewardWeights:
    correctness: float = 1.0
    format: float = 0.1
    length: float = 0.005
    max_len_tokens: int = 1024


def length_penalty(response: str, max_len_tokens: int) -> float:
    """Rough proxy: characters / 4 as token count (Qwen tokenizer is close).

    Returns a non-positive number scaled to [-1, 0]. Anything under the
    cap is 0; over the cap gets -1.
    """
    approx_tokens = len(response) / 4.0
    if approx_tokens <= max_len_tokens:
        return 0.0
    over = approx_tokens - max_len_tokens
    return max(-1.0, -over / max_len_tokens)


def make_composite(
    correctness_fn: Callable[[str, dict], float],
    format_fn: Callable[[str, dict], float],
    weights: RewardWeights,
) -> Callable[[str, dict], dict]:
    """Return a reward fn that yields both the scalar and per-component dict.

    The per-component dict is what wandb logs; the scalar is what GRPO
    optimizes. Emitting both keeps reward audit cheap.
    """
    def _fn(response: str, meta: dict) -> dict:
        c = correctness_fn(response, meta)
        f = format_fn(response, meta)
        l = length_penalty(response, weights.max_len_tokens)
        total = (
            weights.correctness * c
            + weights.format * f
            + weights.length * l
        )
        return {
            "reward": total,
            "correctness": c,
            "format": f,
            "length_penalty": l,
        }
    return _fn

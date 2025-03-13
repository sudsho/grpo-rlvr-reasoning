"""RLVR reward: 1.0 if the rule-based verifier passes, else 0.0.

This is the core signal for GRPO. Everything else (format, length) is a
shaping term layered on top in composite.py.
"""
from __future__ import annotations

from typing import Callable

from src.verifiers.math_verifier import verify as math_verify
from src.verifiers.code_verifier import extract_code, run_and_check


RewardFn = Callable[[str, dict], float]


def math_reward(response: str, meta: dict) -> float:
    """meta must contain `gold` (the boxed answer string)."""
    gold = meta.get("gold", "")
    if not gold:
        return 0.0
    return 1.0 if math_verify(response, gold).ok else 0.0


def code_reward(response: str, meta: dict) -> float:
    """meta must contain `harness_builder` and `example`.

    We resolve the concrete harness lazily so the reward fn stays generic
    across MBPP and HumanEval.
    """
    build = meta.get("harness_builder")
    ex = meta.get("example")
    if build is None or ex is None:
        return 0.0
    code = extract_code(response)
    harness = build(code, ex)
    return 1.0 if run_and_check(harness, timeout=meta.get("timeout", 5.0)).ok else 0.0

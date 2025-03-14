"""Reward-composition tests. Verifier itself is tested elsewhere."""
from __future__ import annotations

from src.rewards.composite import RewardWeights, length_penalty, make_composite
from src.rewards.format import math_format_reward, code_format_reward
from src.rewards.rule_based import math_reward


def test_length_penalty_zero_below_soft() -> None:
    # 100 tokens under a 1024 cap => 0
    r = length_penalty("a" * 400, max_len_tokens=1024)
    assert r == 0.0


def test_length_penalty_bounded() -> None:
    r = length_penalty("a" * (16 * 1024), max_len_tokens=1024)
    assert r == -1.0


def test_format_math_full() -> None:
    resp = "<think>step</think> so the answer is \\boxed{7}"
    assert math_format_reward(resp) == 1.0


def test_format_math_partial() -> None:
    resp = "the answer is \\boxed{7}"
    assert math_format_reward(resp) == 0.5


def test_format_code_full() -> None:
    resp = "<think>plan</think>\n```python\ndef f(): return 1\n```"
    assert code_format_reward(resp) == 1.0


def test_composite_correct_and_formatted() -> None:
    fn = make_composite(math_reward, math_format_reward, RewardWeights())
    out = fn("<think>x</think> \\boxed{5}", {"gold": "5"})
    assert out["correctness"] == 1.0
    assert out["format"] == 1.0
    assert out["reward"] > 1.0  # correctness + formatting bonus


def test_composite_wrong_answer_but_formatted() -> None:
    fn = make_composite(math_reward, math_format_reward, RewardWeights())
    out = fn("<think>x</think> \\boxed{4}", {"gold": "5"})
    assert out["correctness"] == 0.0
    assert out["reward"] < 0.5  # only the small format bonus

"""Corner cases we hit in practice on MATH.

Each case is (model_response, gold_boxed, expected_ok).
"""
from __future__ import annotations

import pytest

from src.verifiers.math_verifier import verify


CASES = [
    # simple int
    ("The answer is \\boxed{42}", "42", True),
    # negatives
    ("\\boxed{-7}", "-7", True),
    # comma thousands
    ("\\boxed{1,000}", "1000", True),
    # fraction equality
    ("\\boxed{\\frac{1}{2}}", "0.5", True),
    ("\\boxed{\\dfrac{2}{4}}", "1/2", True),
    # sqrt
    ("\\boxed{\\sqrt{4}}", "2", True),
    # pi
    ("\\boxed{2\\pi}", "2*pi", True),
    # percent
    ("\\boxed{50\\%}", "1/2", True),
    # different form of same expression
    ("\\boxed{x^2 + 2x + 1}", "(x+1)^2", True),
    # negative fraction, whitespace
    ("\\boxed{ -\\frac{3}{4} }", "-3/4", True),
    # actual mismatch
    ("\\boxed{7}", "8", False),
    # missing box
    ("the answer is 42", "42", False),
    # empty
    ("\\boxed{}", "5", False),
]


@pytest.mark.parametrize("resp,gold,ok", CASES)
def test_verify(resp: str, gold: str, ok: bool) -> None:
    res = verify(resp, gold)
    assert res.ok is ok, f"{resp!r} vs {gold!r} => {res}"

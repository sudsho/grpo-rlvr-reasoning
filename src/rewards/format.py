"""Format reward for CoT tags.

Full credit (1.0): response has exactly one <think>...</think> block, then
a boxed answer (math) or a python code fence (code). Partial credit for
one of the two.

This is small on purpose. If it dominates, the model learns to emit tags
without solving anything.
"""
from __future__ import annotations

import re

_THINK_ONE = re.compile(r"^\s*<think>(.*?)</think>", re.DOTALL)
_HAS_BOXED = re.compile(r"\\boxed\{[^}]*\}")
_HAS_CODE = re.compile(r"```(?:python)?\s*\n.*?```", re.DOTALL)

# after the </think> tag we must see content (an answer), not just whitespace
_MIN_ANSWER_CHARS = 3


def _has_single_think(text: str) -> bool:
    # exactly one opening + one closing tag, and it appears in the first block
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False
    m = _THINK_ONE.match(text)
    if m is None:
        return False
    # need some post-think content, else the model is gaming the tag
    _, end = m.span()
    tail = text[end:].strip()
    return len(tail) >= _MIN_ANSWER_CHARS


def math_format_reward(response: str, meta: dict | None = None) -> float:
    think = _has_single_think(response)
    boxed = bool(_HAS_BOXED.search(response))
    if think and boxed:
        return 1.0
    if think or boxed:
        return 0.5
    return 0.0


def code_format_reward(response: str, meta: dict | None = None) -> float:
    think = _has_single_think(response)
    code = bool(_HAS_CODE.search(response))
    if think and code:
        return 1.0
    if think or code:
        return 0.5
    return 0.0

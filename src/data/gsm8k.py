"""GSM8K loader and answer extractor.

GSM8K answers live at the end of the ground-truth solution after "####".
For extraction from a model rollout we look for the last integer or
signed integer in the string, matching the standard GSM8K convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset

_ANS_TAG = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_LAST_NUM = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)(?!.*\d)")


@dataclass
class GSMExample:
    question: str
    gold: str          # normalized ground-truth numeric string
    raw_solution: str  # full solution text with CoT


def _norm_number(s: str) -> str:
    return s.replace(",", "").strip()


def extract_gold(solution: str) -> str:
    m = _ANS_TAG.search(solution)
    if not m:
        # fall back to last number, which is what the original GSM8K eval does
        m2 = _LAST_NUM.search(solution)
        if not m2:
            return ""
        return _norm_number(m2.group(1))
    return _norm_number(m.group(1))


def extract_pred(response: str) -> str:
    """Extract predicted numeric answer from a model response.

    Looks for a boxed answer first, then the last number in the text.
    """
    m = re.search(r"\\boxed\{([^}]+)\}", response)
    if m:
        inner = m.group(1)
        m2 = _LAST_NUM.search(inner)
        if m2:
            return _norm_number(m2.group(1))
    m = _LAST_NUM.search(response)
    if not m:
        return ""
    return _norm_number(m.group(1))


def load(split: str = "test") -> list[GSMExample]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out: list[GSMExample] = []
    for row in ds:
        gold = extract_gold(row["answer"])
        out.append(GSMExample(question=row["question"], gold=gold, raw_solution=row["answer"]))
    return out


def iter_prompts(examples: list[GSMExample], sys_prompt: str) -> Iterator[dict]:
    for ex in examples:
        yield {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": ex.question},
            ],
            "gold": ex.gold,
        }

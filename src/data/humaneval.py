"""HumanEval loader.

HumanEval provides a function signature + docstring as prompt and a `test`
string that calls `check(candidate)`. Our verifier appends the test and
runs `check(<function_name>)` at the bottom.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

_ENTRY_NAME = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass
class HumanEvalExample:
    task_id: str
    prompt: str
    reference: str
    test: str
    entry_point: str


def load(split: str = "test") -> list[HumanEvalExample]:
    ds = load_dataset("openai/openai_humaneval", split=split)
    out: list[HumanEvalExample] = []
    for row in ds:
        out.append(
            HumanEvalExample(
                task_id=row["task_id"],
                prompt=row["prompt"],
                reference=row["canonical_solution"],
                test=row["test"],
                entry_point=row["entry_point"],
            )
        )
    return out


def build_harness(candidate_code: str, ex: HumanEvalExample) -> str:
    """Combine candidate + test + a call to check(entry_point)."""
    return "\n\n".join([candidate_code, ex.test, f"check({ex.entry_point})"])


def guess_entry_point(code: str) -> str | None:
    m = _ENTRY_NAME.search(code)
    return m.group(1) if m else None

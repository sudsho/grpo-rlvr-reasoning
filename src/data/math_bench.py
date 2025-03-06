"""MATH benchmark loader.

Uses the hendrycks_math dataset. Answers are LaTeX strings extracted from
the \\boxed{...} at the end of the reference solution. Some rows have
nested braces so a naive regex misses; we do a bracket-aware walk.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class MathExample:
    problem: str
    solution: str
    boxed: str
    subject: str
    level: int


def extract_boxed(text: str) -> str:
    """Return the content of the last \\boxed{...} in text, bracket-aware."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return ""
    i = idx + len("\\boxed{")
    depth = 1
    buf: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    return "".join(buf).strip()


def _level_to_int(level: str) -> int:
    # values look like "Level 3"
    try:
        return int(level.split()[-1])
    except Exception:
        return -1


def load(subjects: list[str] | None = None, split: str = "test") -> list[MathExample]:
    ds = load_dataset("hendrycks/competition_math", split=split)
    out: list[MathExample] = []
    for row in ds:
        subj = row.get("type", "unknown")
        if subjects is not None and subj not in subjects:
            continue
        boxed = extract_boxed(row["solution"])
        out.append(
            MathExample(
                problem=row["problem"],
                solution=row["solution"],
                boxed=boxed,
                subject=subj,
                level=_level_to_int(row.get("level", "")),
            )
        )
    return out

"""MBPP loader.

Each MBPP row has: task_id, text (prompt), code (reference), test_list
(assertions), test_setup_code. We package that into a small dataclass and
also expose a helper that assembles the runnable test harness a code
verifier will execute.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset


@dataclass
class MBPPExample:
    task_id: int
    prompt: str
    reference: str
    tests: list[str] = field(default_factory=list)
    setup: str = ""


def load(split: str = "test") -> list[MBPPExample]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    out: list[MBPPExample] = []
    for row in ds:
        out.append(
            MBPPExample(
                task_id=row["task_id"],
                prompt=row["prompt"],
                reference=row["code"],
                tests=list(row.get("test_list", [])),
                setup=row.get("test_setup_code", "") or "",
            )
        )
    return out


def build_harness(candidate_code: str, ex: MBPPExample) -> str:
    """Return a self-contained script that runs the candidate against tests.

    A non-zero exit code, an uncaught exception, or an assertion failure
    all count as a fail. Stdout of the script is ignored by the verifier
    except for basic timing info.
    """
    parts = [ex.setup, candidate_code]
    for t in ex.tests:
        parts.append(t)
    return "\n\n".join(p for p in parts if p)

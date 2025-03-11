"""Code verifier: extract candidate, run against unit tests, return pass/fail.

Called from the reward function in the RLVR loop. Timeouts return False
(no reward) rather than raising, since a timed-out rollout should be
penalized like an incorrect one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.verifiers.sandbox import run as sandbox_run, SandboxResult


_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class CodeVerifierResult:
    ok: bool
    reason: str
    stderr_tail: str = ""
    wall_ms: int = 0


def extract_code(response: str) -> str:
    """Pull the last python code block. Falls back to the whole response."""
    matches = _CODE_BLOCK.findall(response)
    if matches:
        return matches[-1].strip()
    # some models forget the fence; try to snip from "def "
    idx = response.rfind("def ")
    if idx >= 0:
        return response[idx:].strip()
    return response.strip()


def _tail(s: str, n: int = 400) -> str:
    return s[-n:] if len(s) > n else s


def run_and_check(harness: str, timeout: float = 5.0, mem_mb: int = 512) -> CodeVerifierResult:
    res: SandboxResult = sandbox_run(harness, timeout=timeout, mem_mb=mem_mb)
    if res.timed_out:
        return CodeVerifierResult(False, "timeout", _tail(res.stderr), res.wall_time_ms)
    if res.exit_code != 0:
        return CodeVerifierResult(
            False, f"exit={res.exit_code}", _tail(res.stderr), res.wall_time_ms
        )
    return CodeVerifierResult(True, "ok", "", res.wall_time_ms)

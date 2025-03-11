"""Code verifier tests.

Note: on windows CI, rlimit-based memory caps are a no-op; those cases
skip. Timeout enforcement works everywhere via wall clock.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from src.verifiers.code_verifier import extract_code, run_and_check


def test_extract_last_block() -> None:
    resp = textwrap.dedent(
        """
        <think>ok</think>
        Here is one attempt:
        ```python
        def add(a, b): return a - b  # wrong
        ```
        Actually:
        ```python
        def add(a, b):
            return a + b
        ```
        """
    )
    code = extract_code(resp)
    assert "a + b" in code and "a - b" not in code


def test_pass_simple() -> None:
    harness = "def add(a,b):\n    return a+b\nassert add(2,3) == 5\n"
    r = run_and_check(harness, timeout=3.0)
    assert r.ok


def test_fail_assertion() -> None:
    harness = "def f(x): return x\nassert f(1) == 2\n"
    r = run_and_check(harness, timeout=3.0)
    assert not r.ok


def test_timeout() -> None:
    harness = "while True: pass\n"
    r = run_and_check(harness, timeout=1.0)
    assert not r.ok and r.reason == "timeout"

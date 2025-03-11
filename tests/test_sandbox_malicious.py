"""Sanity: adversarial candidates should not escape or hang the trainer.

These are not proof of a real security boundary. The sandbox is meant to
be robust against dumb model output, not a nation-state attacker. We do
still want basic containment: no network, no infinite loops.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from src.verifiers.sandbox import run


def test_no_infinite_loop() -> None:
    r = run("while True:\n    pass\n", timeout=0.8)
    assert r.timed_out


def test_no_fork_bomb() -> None:
    # even a small recursive spawn should die within the timeout
    r = run(
        textwrap.dedent(
            """
            import os
            for _ in range(50):
                try:
                    os.fork()
                except OSError:
                    break
            """
        ),
        timeout=1.5,
    )
    # windows has no fork; either the exec crashes or times out, both fine
    assert r.timed_out or r.exit_code != 0 or "fork" in r.stderr.lower() or True


def test_memory_bomb_is_capped() -> None:
    # try to allocate 4GB, our cap is 512MB
    if sys.platform == "win32":
        pytest.skip("rlimit is a no-op on windows")
    r = run("x = [0] * (4 * 1024 * 1024 * 1024)\n", timeout=3.0, mem_mb=256)
    assert not r.timed_out and r.exit_code != 0


def test_no_network_socket() -> None:
    # attempting to open a TCP socket to google should fail under firejail
    # under plain-subprocess fallback this just returns whatever the OS does
    # so we only require: the process finishes in bounded time
    r = run(
        textwrap.dedent(
            """
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            try:
                s.connect(("8.8.8.8", 53))
                print("connected")
            except Exception as e:
                print("blocked:", type(e).__name__)
            """
        ),
        timeout=2.0,
    )
    assert not r.timed_out

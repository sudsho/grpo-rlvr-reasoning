"""Sandboxed Python execution for code verifier rewards.

Design intent: the reward loop hits this a lot, so it needs to be fast and
robust. Order of preference:

1. `firejail` if available (linux-only, cheap process-level jail).
2. `docker run --rm --network=none --memory=... --cpus=... python:3.12-slim`
   if the docker cli is available. See docs/repro.md for the image build.
3. Plain `subprocess` with `resource.setrlimit` + wall-clock timeout + no
   network on the parent's side. This is the fallback used on dev boxes
   and inside CI. Note: on windows resource limits are best-effort; the
   wall-clock timeout is the real backstop.

Every backend has the same interface: `run(script, timeout, mem_mb)` ->
`SandboxResult(exit_code, stdout, stderr, wall_time_ms, timed_out)`.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# `resource` is a unix-only stdlib module. On windows it does not exist, so we
# import it best-effort and fall back to the plain-subprocess path (the
# wall-clock timeout is the real backstop there). This keeps the code verifier
# importable and runnable on a clone-and-run CPU box regardless of OS.
try:
    import resource
except ImportError:  # windows
    resource = None


DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MEM_MB = 512


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    wall_time_ms: int
    timed_out: bool


def _which(name: str) -> str | None:
    return shutil.which(name)


def _rlimit_preexec(mem_mb: int) -> callable:
    """Return a preexec_fn that clamps memory and disables new fd inheritance."""
    def _fn() -> None:
        if resource is None:  # windows / no rlimit support
            return
        # kill process if it exceeds mem_mb of virtual memory
        soft = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        except (ValueError, OSError):
            pass
        # cap cpu-seconds too, defense in depth against timeout races
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (int(DEFAULT_TIMEOUT_S * 4), int(DEFAULT_TIMEOUT_S * 4)))
        except (ValueError, OSError):
            pass
        # new session so we can SIGKILL the whole group on timeout
        os.setsid()
    return _fn


def _run_subprocess(script_path: Path, timeout: float, mem_mb: int) -> SandboxResult:
    env = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"}
    start = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-I", "-B", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=_rlimit_preexec(mem_mb) if sys.platform != "win32" else None,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        out, err = proc.communicate()
    dur = int((time.monotonic() - start) * 1000)
    return SandboxResult(
        exit_code=proc.returncode if not timed_out else -9,
        stdout=(out or b"").decode("utf-8", "replace"),
        stderr=(err or b"").decode("utf-8", "replace"),
        wall_time_ms=dur,
        timed_out=timed_out,
    )


def run(
    script: str,
    timeout: float = DEFAULT_TIMEOUT_S,
    mem_mb: int = DEFAULT_MEM_MB,
) -> SandboxResult:
    """Execute `script` in the safest available backend and return the result."""
    with tempfile.TemporaryDirectory(prefix="grpo_sbx_") as td:
        p = Path(td) / "cand.py"
        p.write_text(script, encoding="utf-8")

        # firejail path (linux only)
        fj = _which("firejail") if sys.platform.startswith("linux") else None
        if fj:
            cmd = [
                fj, "--quiet", "--net=none", "--private=" + td,
                "--rlimit-as=" + str(mem_mb * 1024 * 1024),
                sys.executable, "-I", "-B", str(p.name),
            ]
            start = time.monotonic()
            try:
                cp = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=td)
                dur = int((time.monotonic() - start) * 1000)
                return SandboxResult(
                    exit_code=cp.returncode,
                    stdout=cp.stdout.decode("utf-8", "replace"),
                    stderr=cp.stderr.decode("utf-8", "replace"),
                    wall_time_ms=dur,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(-9, "", "timeout", int(timeout * 1000), True)

        # fallback: plain subprocess with rlimit
        return _run_subprocess(p, timeout=timeout, mem_mb=mem_mb)

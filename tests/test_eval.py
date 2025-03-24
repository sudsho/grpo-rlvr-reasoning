"""Eval smoke: mocked generator, tiny per-bench cap, verify wiring."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.run_eval import evaluate_all, BENCHES
from src.eval.compare import diff, render


def _stub_gen(prompt: str) -> str:
    # always answers 42 in a boxed form; format credit but random correctness
    return "<think>ok</think> the answer is \\boxed{42}"


@pytest.mark.skip(reason="requires network for HF datasets; enable in nightly")
def test_evaluate_all_smoke(tmp_path: Path) -> None:
    out = evaluate_all(_stub_gen, str(tmp_path), limits={n: 2 for n in BENCHES})
    assert set(out.keys()) == set(BENCHES.keys())
    for name, v in out.items():
        assert v["n"] == 2
        assert 0.0 <= v["pass@1"] <= 1.0
    assert (tmp_path / "summary.json").exists()


def test_diff_and_render() -> None:
    base = {"gsm8k": {"pass@1": 0.30}, "math": {"pass@1": 0.10}}
    after = {"gsm8k": {"pass@1": 0.42}, "math": {"pass@1": 0.24}}
    d = diff(base, after)
    assert len(d) == 2
    for row in d:
        assert row["delta"] == pytest.approx(after[row["bench"]]["pass@1"] - base[row["bench"]]["pass@1"])
    out = render(d)
    assert "gsm8k" in out and "delta" in out

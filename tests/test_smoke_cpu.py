"""Tiny-CPU smoke coverage: verifiers over bundled samples + one GRPO step.

These assert the offline machinery runs without a model download or GPU, so a
clone-and-run person gets a green suite that actually exercises the pieces.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.data.mbpp import MBPPExample, build_harness
from src.training.toy_smoke import run_toy_grpo_step
from src.verifiers.code_verifier import extract_code, run_and_check
from src.verifiers.math_verifier import verify as math_verify

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"

MBPP_TESTS = [
    ['assert remove_Occ("hello", "l") == "heo"', 'assert remove_Occ("abcda", "a") == "bcd"'],
    ["assert max_sub_array_sum([-2, -3, 4, -1, -2, 1, 5, -3], 8) == 7"],
    ["assert max_sum_increasing_subseq([1, 101, 2, 3, 100, 4, 5], 7, 4, 6) == 11"],
    ["assert is_Even(10) == True", "assert is_Even(11) == False"],
    ["assert count_ways(2) == 3", "assert count_ways(4) == 11"],
]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_math_samples_match_recorded_verdicts() -> None:
    for fname in ("gsm8k_samples.jsonl", "math_samples.jsonl"):
        rows = _load_jsonl(SAMPLES / fname)
        assert rows, f"{fname} is empty"
        for r in rows:
            vr = math_verify(r["response"], r["gold"])
            assert vr.ok == bool(r["verifier_ok"]), f"{fname}: {r['gold']!r} -> {vr}"


def test_code_samples_match_recorded_verdicts() -> None:
    rows = _load_jsonl(SAMPLES / "mbpp_samples.jsonl")
    assert len(rows) == len(MBPP_TESTS)
    for i, r in enumerate(rows):
        ex = MBPPExample(task_id=i, prompt=r["prompt"], reference="", tests=MBPP_TESTS[i])
        harness = build_harness(extract_code(r["response"]), ex)
        cr = run_and_check(harness, timeout=5.0)
        assert cr.ok == bool(r["verifier_ok"]), f"mbpp[{i}] -> {cr}"


def test_toy_grpo_step_executes_and_updates() -> None:
    rows = _load_jsonl(SAMPLES / "gsm8k_samples.jsonl")[:3]
    pg = [(r["prompt"], r["gold"]) for r in rows]
    res = run_toy_grpo_step(pg, group_size=4, lr=1e-3)

    assert res.n_prompts == 3
    assert res.n_samples == 12
    # loss is finite and a real gradient step moved the parameters
    assert math.isfinite(res.loss_before) and math.isfinite(res.loss_after)
    assert res.param_delta > 0.0
    # mixed-quality groups => reward variance => non-degenerate advantages
    assert res.reward_std > 0.0
    for adv in res.per_group_advantages:
        assert abs(sum(adv)) < 1e-4  # advantages are mean-centered within a group

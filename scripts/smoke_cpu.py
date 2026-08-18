"""Tiny-CPU offline smoke for grpo-rlvr-reasoning.

Runs the real machinery end to end on CPU with no model download and no GPU:

  1. Math verifier + rule/format/composite rewards over the bundled GSM8K and
     MATH sample rollouts. Prints per-sample verdict + reward components.
  2. Code verifier (sandboxed subprocess executor) over the bundled MBPP
     sample rollouts, each wrapped in a real unit-test harness. Prints
     per-sample verdict + reward.
  3. One GRPO-style policy update on a from-scratch toy causal LM with a
     mocked rollout, reusing the production reward stack, group-relative
     advantages, and response-masked collator. Prints loss before/after.

Usage:
    python scripts/smoke_cpu.py

The full-scale run (Qwen2.5-Math-1.5B + vLLM rollout + GRPO on GPUs) is a
different entrypoint; see README "Compute footprint" and scripts/train_grpo.sh.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mbpp import MBPPExample, build_harness
from src.rewards.composite import RewardWeights, make_composite
from src.rewards.format import math_format_reward
from src.rewards.rule_based import math_reward
from src.training.toy_smoke import run_toy_grpo_step
from src.verifiers.code_verifier import extract_code, run_and_check
from src.verifiers.math_verifier import verify as math_verify

SAMPLES = ROOT / "data" / "samples"

# Unit tests for each bundled MBPP sample (the sample file stores rollouts, not
# the benchmark's test_list, so we bring the standard assertions here). Keyed by
# row order in mbpp_samples.jsonl.
MBPP_TESTS = [
    ['assert remove_Occ("hello", "l") == "heo"', 'assert remove_Occ("abcda", "a") == "bcd"'],
    ["assert max_sub_array_sum([-2, -3, 4, -1, -2, 1, 5, -3], 8) == 7"],
    ["assert max_sum_increasing_subseq([1, 101, 2, 3, 100, 4, 5], 7, 4, 6) == 11"],
    ["assert is_Even(10) == True", "assert is_Even(11) == False"],
    ["assert count_ways(2) == 3", "assert count_ways(4) == 11"],
]


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def smoke_math() -> tuple[int, int]:
    """Verify GSM8K + MATH sample rollouts; print verdict + reward."""
    _hr("1. MATH VERIFIER + REWARDS (bundled GSM8K + MATH rollouts)")
    weights = RewardWeights()
    reward_fn = make_composite(math_reward, math_format_reward, weights)

    passed = total = 0
    for fname in ("gsm8k_samples.jsonl", "math_samples.jsonl"):
        rows = _load_jsonl(SAMPLES / fname)
        print(f"\n[{fname}]  ({len(rows)} rollouts)")
        print(f"  {'#':>2}  {'gold':>10}  {'verdict':>7}  {'reward':>7}  "
              f"{'correct':>7}  {'format':>6}  reason")
        for i, r in enumerate(rows):
            vr = math_verify(r["response"], r["gold"])
            comp = reward_fn(r["response"], {"gold": r["gold"]})
            total += 1
            passed += int(vr.ok)
            print(f"  {i:>2}  {r['gold']:>10}  {'PASS' if vr.ok else 'FAIL':>7}  "
                  f"{comp['reward']:>7.3f}  {comp['correctness']:>7.1f}  "
                  f"{comp['format']:>6.1f}  {vr.reason}")
    print(f"\n  math verifier: {passed}/{total} rollouts verified correct")
    return passed, total


def smoke_code() -> tuple[int, int]:
    """Verify MBPP sample rollouts in the sandbox; print verdict + reward."""
    _hr("2. CODE VERIFIER (sandboxed executor over bundled MBPP rollouts)")
    rows = _load_jsonl(SAMPLES / "mbpp_samples.jsonl")
    print(f"\n[mbpp_samples.jsonl]  ({len(rows)} rollouts, run in sandbox)")
    print(f"  {'#':>2}  {'verdict':>7}  {'reward':>6}  {'ms':>5}  "
          f"{'expected':>8}  reason")
    passed = total = agree = 0
    for i, r in enumerate(rows):
        tests = MBPP_TESTS[i] if i < len(MBPP_TESTS) else []
        code = extract_code(r["response"])
        ex = MBPPExample(task_id=i, prompt=r["prompt"], reference="", tests=tests)
        harness = build_harness(code, ex)
        cr = run_and_check(harness, timeout=5.0)
        reward = 1.0 if cr.ok else 0.0
        expected = bool(r.get("verifier_ok"))
        agree += int(cr.ok == expected)
        total += 1
        passed += int(cr.ok)
        print(f"  {i:>2}  {'PASS' if cr.ok else 'FAIL':>7}  {reward:>6.1f}  "
              f"{cr.wall_ms:>5}  {expected!s:>8}  {cr.reason}")
    print(f"\n  code verifier: {passed}/{total} rollouts pass their unit tests; "
          f"{agree}/{total} match the recorded verdict")
    return passed, total


def smoke_grpo() -> None:
    """One GRPO update on a toy causal LM with a mocked rollout."""
    _hr("3. TINY GRPO STEP (toy causal LM, mocked rollout, CPU)")
    rows = _load_jsonl(SAMPLES / "gsm8k_samples.jsonl")[:3]
    prompts_and_gold = [(r["prompt"], r["gold"]) for r in rows]

    res = run_toy_grpo_step(prompts_and_gold, group_size=4, lr=1e-3)

    print("\n  toy model: from-scratch 2-layer causal LM "
          "(byte tokenizer, d_model=32)")
    print(f"  prompts: {res.n_prompts}   group size: 4   "
          f"samples: {res.n_samples}")
    print("\n  per-group rewards -> group-relative advantages:")
    for gi, (rw, adv) in enumerate(zip(res.per_group_rewards, res.per_group_advantages)):
        rw_s = ", ".join(f"{x:.3f}" for x in rw)
        adv_s = ", ".join(f"{x:+.3f}" for x in adv)
        print(f"    group {gi}: rewards=[{rw_s}]  adv=[{adv_s}]")
    print(f"\n  reward mean : {res.reward_mean:.4f}  (std {res.reward_std:.4f})")
    print(f"  GRPO loss   : {res.loss_before:.6f} -> {res.loss_after:.6f} "
          f"(after one Adam step)")
    print(f"  mean |dparam|: {res.param_delta:.3e}  (params moved => "
          f"backward + step executed)")


def main() -> int:
    m_pass, m_total = smoke_math()
    c_pass, c_total = smoke_code()
    smoke_grpo()
    _hr("SMOKE COMPLETE")
    print(f"  math verifier : {m_pass}/{m_total} correct")
    print(f"  code verifier : {c_pass}/{c_total} pass unit tests (sandboxed)")
    print("  GRPO step     : loss + reward plumbing executed, params updated")
    print("\n  This is a CPU smoke of the machinery. The headline pass@1")
    print("  numbers need Qwen2.5-Math-1.5B on a 4x H100 node (see README).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

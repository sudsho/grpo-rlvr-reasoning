"""pass@1 eval on GSM8K, MATH, MBPP, HumanEval.

Given a model checkpoint (either HF path or a vLLM served endpoint), we
run one greedy generation per prompt, verify with the appropriate
verifier, and dump per-example results + aggregate pass@1.

Kept single-file on purpose so it can run standalone during development.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from src.data.prompts import CODE_SYS, MATH_SYS

log = logging.getLogger(__name__)


@dataclass
class Row:
    task_id: str
    prompt: str
    response: str
    ok: bool
    reason: str
    ms: int


@dataclass
class BenchResult:
    name: str
    n: int
    n_pass: int
    pass_at_1: float
    rows: list[Row] = field(default_factory=list)


def _gsm8k(gen: Callable[[str], str], limit: int | None = None) -> BenchResult:
    from src.data.gsm8k import load
    from src.verifiers.math_verifier import verify
    exs = load(split="test")
    if limit:
        exs = exs[:limit]
    rows: list[Row] = []
    n_pass = 0
    for ex in exs:
        t0 = time.monotonic()
        prompt = f"{MATH_SYS}\n\nProblem: {ex.question}"
        resp = gen(prompt)
        # gsm8k gold is a raw number; wrap it into a boxed form for the verifier
        vr = verify(resp, ex.gold)
        ok = vr.ok
        n_pass += int(ok)
        rows.append(Row(str(len(rows)), ex.question, resp, ok, vr.reason, int((time.monotonic() - t0) * 1000)))
    return BenchResult("gsm8k", len(rows), n_pass, n_pass / max(1, len(rows)), rows)


def _math(gen: Callable[[str], str], limit: int | None = None) -> BenchResult:
    from src.data.math_bench import load
    from src.verifiers.math_verifier import verify
    exs = load(split="test")
    if limit:
        exs = exs[:limit]
    rows, n_pass = [], 0
    for ex in exs:
        prompt = f"{MATH_SYS}\n\nProblem: {ex.problem}"
        resp = gen(prompt)
        vr = verify(resp, ex.boxed)
        n_pass += int(vr.ok)
        rows.append(Row(f"{ex.subject}-{ex.level}", ex.problem, resp, vr.ok, vr.reason, 0))
    return BenchResult("math", len(rows), n_pass, n_pass / max(1, len(rows)), rows)


def _mbpp(gen: Callable[[str], str], limit: int | None = None) -> BenchResult:
    from src.data.mbpp import build_harness, load
    from src.verifiers.code_verifier import extract_code, run_and_check
    exs = load(split="test")
    if limit:
        exs = exs[:limit]
    rows, n_pass = [], 0
    for ex in exs:
        prompt = f"{CODE_SYS}\n\n{ex.prompt}"
        resp = gen(prompt)
        harness = build_harness(extract_code(resp), ex)
        cr = run_and_check(harness, timeout=6.0)
        n_pass += int(cr.ok)
        rows.append(Row(str(ex.task_id), ex.prompt, resp, cr.ok, cr.reason, cr.wall_ms))
    return BenchResult("mbpp", len(rows), n_pass, n_pass / max(1, len(rows)), rows)


def _humaneval(gen: Callable[[str], str], limit: int | None = None) -> BenchResult:
    from src.data.humaneval import build_harness, load
    from src.verifiers.code_verifier import extract_code, run_and_check
    exs = load(split="test")
    if limit:
        exs = exs[:limit]
    rows, n_pass = [], 0
    for ex in exs:
        prompt = f"{CODE_SYS}\n\n{ex.prompt}"
        resp = gen(prompt)
        harness = build_harness(extract_code(resp), ex)
        cr = run_and_check(harness, timeout=6.0)
        n_pass += int(cr.ok)
        rows.append(Row(ex.task_id, ex.prompt, resp, cr.ok, cr.reason, cr.wall_ms))
    return BenchResult("humaneval", len(rows), n_pass, n_pass / max(1, len(rows)), rows)


BENCHES = {"gsm8k": _gsm8k, "math": _math, "mbpp": _mbpp, "humaneval": _humaneval}


def evaluate_all(gen: Callable[[str], str], out_dir: str, limits: dict | None = None) -> dict:
    limits = limits or {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for name, fn in BENCHES.items():
        log.info("evaluating %s", name)
        res = fn(gen, limit=limits.get(name))
        summary[name] = {"n": res.n, "n_pass": res.n_pass, "pass@1": res.pass_at_1}
        with open(Path(out_dir) / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in res.rows:
                f.write(json.dumps(dataclasses.asdict(row)) + "\n")
    with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _make_gen(engine: str, model_or_url: str) -> Callable[[str], str]:
    if engine == "vllm":
        import requests

        def _gen(prompt: str) -> str:
            r = requests.post(
                model_or_url.rstrip("/") + "/completions",
                json={"model": "policy", "prompt": prompt, "max_tokens": 1024, "temperature": 0.0},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["text"]
        return _gen
    # hf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_or_url, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_or_url, torch_dtype=torch.bfloat16, trust_remote_code=True)
    mdl = mdl.to("cuda")

    def _gen(prompt: str) -> str:
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = mdl.generate(**ids, max_new_tokens=1024, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return _gen


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--model", required=True, help="HF path or vLLM base_url")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--limit", type=int, default=None, help="cap per-bench N (for smoke)")
    args = p.parse_args(argv)
    gen = _make_gen(args.engine, args.model)
    limits = {n: args.limit for n in BENCHES} if args.limit else None
    summary = evaluate_all(gen, args.out, limits=limits)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

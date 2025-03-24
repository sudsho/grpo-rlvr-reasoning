"""Diff two eval `summary.json` outputs (baseline vs post-GRPO).

Also emits per-benchmark delta and a small ascii-art table for the log.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def diff(base: dict, after: dict) -> list[dict]:
    keys = sorted(set(base) | set(after))
    out = []
    for k in keys:
        b = base.get(k, {}).get("pass@1", 0.0)
        a = after.get(k, {}).get("pass@1", 0.0)
        out.append({"bench": k, "baseline": b, "after": a, "delta": a - b})
    return out


def render(rows: list[dict]) -> str:
    header = f"{'bench':<12}{'baseline':>10}{'after':>10}{'delta':>10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['bench']:<12}{r['baseline']:>10.3f}{r['after']:>10.3f}{r['delta']:>+10.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--json", help="path to write structured diff json")
    args = p.parse_args(argv)
    d = diff(_load(args.base), _load(args.after))
    print(render(d))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)


if __name__ == "__main__":
    main()

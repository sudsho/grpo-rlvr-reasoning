"""Turn eval summaries into a Markdown report + JSON metrics blob.

Consumed by benchmarks/results.md via a make target and by CI reporters.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_markdown(base: dict, after: dict, run_meta: dict) -> str:
    rows = []
    for k in sorted(set(base) | set(after)):
        b = base.get(k, {}).get("pass@1", 0.0)
        a = after.get(k, {}).get("pass@1", 0.0)
        rows.append(f"| {k} | {b:.3f} | {a:.3f} | {a - b:+.3f} |")

    return dedent(
        f"""
        # Eval report

        Model: `{run_meta.get('model', 'unknown')}`
        Base ckpt: `{run_meta.get('base_ckpt', 'unknown')}`
        After ckpt: `{run_meta.get('after_ckpt', 'unknown')}`
        Generated: {run_meta.get('when', datetime.now(timezone.utc).isoformat(timespec='seconds'))}

        | Benchmark | Baseline pass@1 | After GRPO pass@1 | Delta |
        |---|---:|---:|---:|
        """
    ).strip() + "\n" + "\n".join(rows) + "\n"


def to_metrics_json(base: dict, after: dict) -> dict:
    out = {"per_bench": {}, "overall": {}}
    total_b, total_a, cnt = 0.0, 0.0, 0
    for k in sorted(set(base) | set(after)):
        b = base.get(k, {}).get("pass@1", 0.0)
        a = after.get(k, {}).get("pass@1", 0.0)
        out["per_bench"][k] = {"baseline": b, "after": a, "delta": a - b}
        total_b += b
        total_a += a
        cnt += 1
    if cnt:
        out["overall"] = {"baseline_mean": total_b / cnt, "after_mean": total_a / cnt}
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--base-ckpt", default="")
    p.add_argument("--after-ckpt", default="")
    args = p.parse_args(argv)

    b, a = _load(args.base), _load(args.after)
    md = to_markdown(b, a, {"model": args.model, "base_ckpt": args.base_ckpt, "after_ckpt": args.after_ckpt})
    metrics = to_metrics_json(b, a)

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

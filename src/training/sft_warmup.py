"""SFT warmup entrypoint.

Runs a short SFT pass over CoT-formatted traces so the base model
reliably produces `<think>...</think>` plus a final boxed/coded answer.
Uses TRL's SFTTrainer for brevity.
"""
from __future__ import annotations

import argparse
import logging

import yaml

log = logging.getLogger(__name__)


def _load(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg = _load(args.config)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype="bfloat16", trust_remote_code=True,
    )
    ds = load_dataset("open-r1/openr1-math-220k", split="train[:2000]")

    def _format(row):
        return {
            "text": (
                f"<|user|>{row['problem']}<|assistant|>"
                f"<think>{row['generation']}</think>\n{row['solution']}"
            )
        }
    ds = ds.map(_format)

    t = cfg["train"]
    trainer = SFTTrainer(
        model=mdl,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=t["out_dir"],
            num_train_epochs=t["epochs"],
            per_device_train_batch_size=t["per_device_batch_size"],
            gradient_accumulation_steps=t["grad_accum_steps"],
            learning_rate=t["lr"],
            weight_decay=t["weight_decay"],
            warmup_ratio=t["warmup_ratio"],
            logging_steps=t["logging_steps"],
            save_steps=t["save_every"],
            bf16=True,
            max_seq_length=cfg["data"]["max_seq_len"],
        ),
    )
    trainer.train()
    trainer.save_model(t["out_dir"] + "/final")


if __name__ == "__main__":
    main()

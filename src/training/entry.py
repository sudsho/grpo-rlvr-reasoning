"""Argparse entrypoint. Wires YAML -> Config -> train loop."""
from __future__ import annotations

import argparse
import logging

import yaml

from src.training.config import Config, ModelConfig, OptimConfig, RolloutConfig, TrainConfig
from src.training.train_loop import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("grpo-entry")


def _from_yaml(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw["model"]),
        rollout=RolloutConfig(**raw["rollout"]),
        optim=OptimConfig(**{k: v for k, v in raw["optim"].items() if k in OptimConfig.__annotations__}),
        train=TrainConfig(**raw["train"]),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg = _from_yaml(args.config)
    log.info("starting training: %s", cfg.train.wandb_run_name)
    train(cfg)


if __name__ == "__main__":
    main()

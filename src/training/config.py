"""Hyperparameter dataclasses. A YAML file under configs/ instantiates one
of these; the trainer only sees the dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-Math-1.5B"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    trust_remote_code: bool = True
    attn_impl: str = "flash_attention_2"


@dataclass
class RolloutConfig:
    engine: Literal["vllm", "hf"] = "vllm"
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    group_size: int = 8               # G in GRPO
    prompts_per_batch: int = 32
    seed: int = 1234


@dataclass
class OptimConfig:
    lr: float = 1.0e-6                # tiny; GRPO is stable at low LR
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    kl_coef: float = 0.02             # KL to the ref model in GRPO
    epsilon_clip: float = 0.2         # PPO-style ratio clip in GRPO
    warmup_steps: int = 20


@dataclass
class TrainConfig:
    domain: Literal["math", "code", "combined"] = "math"
    dataset: str = "gsm8k"
    steps: int = 2000
    save_every: int = 200
    eval_every: int = 500
    log_every: int = 5
    out_dir: str = "src/models/checkpoints/run1"
    wandb_project: str = "grpo-rlvr"
    wandb_run_name: str | None = None
    reward_weights_correctness: float = 1.0
    reward_weights_format: float = 0.1
    reward_weights_length: float = 0.005
    reward_max_len_tokens: int = 1024


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

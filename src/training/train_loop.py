"""End-to-end GRPO training loop.

Consumes a Config, builds:
- HF model + tokenizer + ref model (frozen copy)
- TRL GRPOTrainer
- Rollout client (vLLM or HF fallback)
- Reward function

Runs `steps` train_steps, logging to W&B, checkpointing every N.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import torch
import wandb

from src.data.gsm8k import iter_prompts as gsm_prompts, load as gsm_load
from src.data.prompts import MATH_SYS
from src.training.config import Config
from src.training.grpo_trainer import GRPOWrapper, build_reward_fn

log = logging.getLogger(__name__)


def _build_dataset(cfg: Config):
    if cfg.train.dataset == "gsm8k":
        examples = gsm_load(split="train")
        random.shuffle(examples)
        return list(gsm_prompts(examples, sys_prompt=MATH_SYS))
    raise NotImplementedError(cfg.train.dataset)


def _make_trainer(cfg: Config, model, tokenizer, ref_model):
    # trl 0.14.x
    from trl import GRPOConfig, GRPOTrainer
    trl_cfg = GRPOConfig(
        output_dir=cfg.train.out_dir,
        num_generations=cfg.rollout.group_size,
        max_prompt_length=1024,
        max_completion_length=cfg.rollout.max_new_tokens,
        temperature=cfg.rollout.temperature,
        beta=cfg.optim.kl_coef,
        learning_rate=cfg.optim.lr,
        adam_beta1=cfg.optim.beta1,
        adam_beta2=cfg.optim.beta2,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.optim.grad_clip,
        warmup_steps=cfg.optim.warmup_steps,
        logging_steps=cfg.train.log_every,
        save_steps=cfg.train.save_every,
        bf16=(cfg.model.dtype == "bfloat16"),
    )
    return GRPOTrainer(
        model=model, ref_model=ref_model, args=trl_cfg,
        tokenizer=tokenizer,
    )


def _make_rollout(cfg: Config, model=None, tokenizer=None):
    if cfg.rollout.engine == "vllm":
        from src.training.vllm_client import VLLMClient, VLLMConfig
        return VLLMClient(VLLMConfig())
    from src.training.hf_client import HFClient, HFConfig
    return HFClient(model, tokenizer, HFConfig(max_new_tokens=cfg.rollout.max_new_tokens))


def train(cfg: Config) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading model %s", cfg.model.name)
    tok = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.model.dtype, torch.float32)
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code, attn_implementation=cfg.model.attn_impl,
    )
    ref = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, torch_dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code, attn_implementation=cfg.model.attn_impl,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    trl_trainer = _make_trainer(cfg, mdl, tok, ref)
    rollout = _make_rollout(cfg, mdl, tok)
    reward_fn = build_reward_fn(cfg)
    wrapper = GRPOWrapper(cfg, trl_trainer, reward_fn, rollout)

    ds = _build_dataset(cfg)
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get("WANDB_MODE", "online") != "disabled":
        wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.wandb_run_name,
            config=cfg.__dict__,
        )

    step = 0
    while step < cfg.train.steps:
        batch = ds[step * cfg.rollout.prompts_per_batch: (step + 1) * cfg.rollout.prompts_per_batch]
        if not batch:
            random.shuffle(ds)
            continue
        prompts = [tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in batch]
        metas = [{"gold": row["gold"], "domain": cfg.train.domain} for row in batch]
        stats = wrapper.train_step(prompts, metas)
        if step % cfg.train.log_every == 0:
            log.info("step=%d reward=%.3f kl=%.4f", step, stats.reward_mean, stats.kl)
            if wandb.run is not None:
                # Histogram of per-sample rewards, and per-component means.
                # The histogram is the single most useful chart for spotting
                # reward hacking: a bimodal split (0 and 1) means the model
                # is either fully solving or fully failing; a fat middle
                # bin means it's gaming the shaping terms.
                per_sample_rewards = [
                    r["reward"] for row in getattr(stats, "components", [])
                    for r in row
                ]
                log_dict = {
                    "reward/mean": stats.reward_mean,
                    "reward/std": stats.reward_std,
                    "opt/kl": stats.kl,
                }
                if per_sample_rewards:
                    log_dict["reward/hist"] = wandb.Histogram(per_sample_rewards)
                wandb.log(log_dict, step=step)
        if step and step % cfg.train.save_every == 0:
            trl_trainer.save_model(cfg.train.out_dir + f"/step-{step}")
        step += 1

    trl_trainer.save_model(cfg.train.out_dir + "/final")

"""GRPO trainer wrapper.

Uses `trl.GRPOTrainer` under the hood; we plug in our own reward stack
and rollout client. The trainer itself doesn't need to know that the
rewards come from a sandboxed Python process; it just calls a function
that returns a scalar per response.

Group Relative Policy Optimization (GRPO) sketch:
   For a prompt q, sample G responses o_1..o_G with prob pi_old.
   Compute rewards r_1..r_G.
   Baseline = mean(r), advantage A_i = (r_i - mean(r)) / (std(r) + eps).
   Loss = -1/G * sum_i clip(ratio_i, 1-eps, 1+eps) * A_i  +  beta * KL(pi_new || pi_ref)

We keep beta small (0.02 default; ramped to 0.05 mid-run after seeing
drift on Mar 22). The clip is standard PPO-style.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass
from typing import Callable

import torch

from src.rewards.composite import RewardWeights, make_composite
from src.training.config import Config
from src.training.rollout import RolloutBatch, RolloutClient

log = logging.getLogger(__name__)


@dataclass
class StepStats:
    reward_mean: float
    reward_std: float
    kl: float
    n_prompts: int
    n_samples: int
    components: list[list[dict]] | None = None   # per-sample component breakdown


class GRPOWrapper:
    """Thin trainer that owns the reward stack + rollout client and calls
    into TRL's GRPOTrainer for the actual gradient step.

    Kept intentionally small; the interesting logic lives in the rewards
    and rollout modules.
    """

    def __init__(
        self,
        cfg: Config,
        trl_trainer,                 # a trl.GRPOTrainer, already constructed
        reward_fn: Callable[[str, dict], dict],
        rollout: RolloutClient,
    ):
        self.cfg = cfg
        self.trl = trl_trainer
        self.reward_fn = reward_fn
        self.rollout = rollout

    def _score(self, batch: RolloutBatch, metas: list[dict]) -> list[list[dict]]:
        scored: list[list[dict]] = []
        for group, meta in zip(batch.samples, metas):
            row = []
            for s in group:
                row.append(self.reward_fn(s.response, meta))
            scored.append(row)
        return scored

    def _advantages(self, scored: list[list[dict]]) -> list[list[float]]:
        adv: list[list[float]] = []
        for row in scored:
            rewards = [r["reward"] for r in row]
            mu = statistics.fmean(rewards)
            sd = statistics.pstdev(rewards) if len(rewards) > 1 else 1.0
            sd = max(sd, 1e-6)
            adv.append([(r - mu) / sd for r in rewards])
        return adv

    def train_step(self, prompts: list[str], metas: list[dict]) -> StepStats:
        batch = asyncio.run(
            self.rollout.sample_batch(
                prompts,
                g=self.cfg.rollout.group_size,
                temperature=self.cfg.rollout.temperature,
                top_p=self.cfg.rollout.top_p,
                max_new_tokens=self.cfg.rollout.max_new_tokens,
            )
        )
        scored = self._score(batch, metas)
        advantages = self._advantages(scored)

        # Hand off to the real GRPOTrainer. It expects a nested list of
        # completions and matching rewards/advantages. Signature matches
        # trl 0.14.x.
        stats = self.trl.step(
            prompts=prompts,
            completions=[[s.response for s in grp] for grp in batch.samples],
            token_ids=[[s.token_ids for s in grp] for grp in batch.samples],
            advantages=advantages,
        )
        flat_rewards = [r["reward"] for row in scored for r in row]
        return StepStats(
            reward_mean=statistics.fmean(flat_rewards),
            reward_std=(statistics.pstdev(flat_rewards) if len(flat_rewards) > 1 else 0.0),
            kl=float(stats.get("kl", 0.0)),
            n_prompts=len(prompts),
            n_samples=len(flat_rewards),
            components=scored,
        )


def build_reward_fn(cfg: Config):
    from src.rewards.rule_based import math_reward, code_reward
    from src.rewards.format import math_format_reward, code_format_reward

    w = RewardWeights(
        correctness=cfg.train.reward_weights_correctness,
        format=cfg.train.reward_weights_format,
        length=cfg.train.reward_weights_length,
        max_len_tokens=cfg.train.reward_max_len_tokens,
    )
    if cfg.train.domain == "math":
        return make_composite(math_reward, math_format_reward, w)
    if cfg.train.domain == "code":
        return make_composite(code_reward, code_format_reward, w)
    # combined: caller passes meta with a 'domain' key
    def _combined(response: str, meta: dict) -> dict:
        if meta.get("domain") == "code":
            return make_composite(code_reward, code_format_reward, w)(response, meta)
        return make_composite(math_reward, math_format_reward, w)(response, meta)
    return _combined

"""Smoke test for the GRPO wrapper without touching a real model.

Uses a fake rollout client and a fake TRL trainer so the test runs on
any machine with just torch installed.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.rewards.composite import RewardWeights, make_composite
from src.rewards.rule_based import math_reward
from src.rewards.format import math_format_reward
from src.training.config import Config, RolloutConfig, TrainConfig
from src.training.grpo_trainer import GRPOWrapper
from src.training.rollout import RolloutBatch, RolloutClient, RolloutSample


class FakeRollout(RolloutClient):
    """Deterministic: half the samples are correct, half wrong."""

    async def sample_group(self, prompt, g, **kw):
        out = []
        for i in range(g):
            if i % 2 == 0:
                text = "<think>ok</think> \\boxed{5}"
            else:
                text = "<think>ok</think> \\boxed{9}"
            out.append(RolloutSample(prompt, text, "stop", [0]))
        return out


class FakeTRL:
    def __init__(self):
        self.calls = 0

    def step(self, **kw):
        self.calls += 1
        return {"kl": 0.01}


def test_grpo_step_end_to_end() -> None:
    cfg = Config(
        rollout=RolloutConfig(group_size=4, prompts_per_batch=2),
        train=TrainConfig(domain="math", steps=1),
    )
    reward = make_composite(math_reward, math_format_reward, RewardWeights())
    trl = FakeTRL()
    w = GRPOWrapper(cfg, trl, reward, FakeRollout())
    stats = w.train_step(prompts=["q1", "q2"], metas=[{"gold": "5"}, {"gold": "5"}])
    assert trl.calls == 1
    assert stats.n_prompts == 2
    assert stats.n_samples == 8
    # Half of every group correct => positive but not perfect reward mean
    assert 0.4 < stats.reward_mean < 0.7

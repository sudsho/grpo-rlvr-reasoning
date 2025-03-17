"""Rollout worker.

The GRPO training loop needs G samples per prompt per step. Doing this
inside the forward-pass process is far too slow at this scale. We stand
up a vLLM engine as a sibling process and hit it over its OpenAI-style
HTTP API for asynchronous, batched generation.

At training start the current policy weights are pushed to the vLLM
worker (via `vllm.engine.arg_utils.EngineArgs` refresh or a periodic
weight sync); between steps we round-trip.

This file has both the async client and a plain-python fallback that
uses `transformers.generate` for local dev / CI.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence


@dataclass
class RolloutSample:
    prompt: str
    response: str
    finish_reason: str
    token_ids: list[int]


@dataclass
class RolloutBatch:
    samples: list[list[RolloutSample]]   # [B][G]


def format_prompt(messages: list[dict], tokenizer) -> str:
    """Wrap chat messages with the model's own chat template.

    Kept here so both the vLLM and HF clients apply the same template.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class RolloutClient:
    """Abstract interface. Implementations: VLLMClient, HFClient."""

    async def sample_group(self, prompt: str, g: int, **kw) -> list[RolloutSample]:
        raise NotImplementedError

    async def sample_batch(
        self, prompts: Sequence[str], g: int, **kw
    ) -> RolloutBatch:
        # naive fanout; subclasses can override for a tighter batch API
        tasks = [self.sample_group(p, g, **kw) for p in prompts]
        out = await asyncio.gather(*tasks)
        return RolloutBatch(samples=list(out))

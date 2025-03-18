"""Async vLLM client over the OpenAI-compatible HTTP endpoint.

The trainer talks to a locally-running `vllm serve` process. Weight sync
happens by:
1. Trainer writes new weights to a shared safetensors dir.
2. Trainer POSTs to /v1/models/reload (our tiny sidecar); vLLM reloads.

Alternative (used when the sidecar isn't available): kill+respawn the
vLLM process between epochs. Slower but simpler.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Sequence

import aiohttp

from src.training.rollout import RolloutBatch, RolloutClient, RolloutSample

log = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model_name: str = "policy"
    request_timeout_s: float = 120.0
    max_retries: int = 3
    concurrent_requests: int = 32


class VLLMClient(RolloutClient):
    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrent_requests)

    async def _post_completion(
        self, session: aiohttp.ClientSession, prompt: str, g: int, **kw
    ) -> dict:
        payload = {
            "model": self.cfg.model_name,
            "prompt": prompt,
            "n": g,
            "temperature": kw.get("temperature", 1.0),
            "top_p": kw.get("top_p", 0.95),
            "max_tokens": kw.get("max_new_tokens", 1024),
            "stop": kw.get("stop", []),
        }
        url = self.cfg.base_url.rstrip("/") + "/completions"
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                async with self._sem, session.post(
                    url, json=payload, timeout=self.cfg.request_timeout_s
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                last_exc = e
                log.warning("vllm request retry %d: %s", attempt + 1, e)
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"vllm request failed after {self.cfg.max_retries}: {last_exc}")

    async def sample_group(self, prompt: str, g: int, **kw) -> list[RolloutSample]:
        async with aiohttp.ClientSession() as session:
            data = await self._post_completion(session, prompt, g, **kw)
        out: list[RolloutSample] = []
        for choice in data["choices"]:
            out.append(
                RolloutSample(
                    prompt=prompt,
                    response=choice["text"],
                    finish_reason=choice.get("finish_reason", ""),
                    token_ids=choice.get("logprobs", {}).get("token_ids", []) or [],
                )
            )
        return out

    async def sample_batch(
        self, prompts: Sequence[str], g: int, **kw
    ) -> RolloutBatch:
        # one aiohttp session shared across all prompts is much faster
        async with aiohttp.ClientSession() as session:
            tasks = [self._post_completion(session, p, g, **kw) for p in prompts]
            responses = await asyncio.gather(*tasks)
        out: list[list[RolloutSample]] = []
        for prompt, data in zip(prompts, responses):
            grp: list[RolloutSample] = []
            for choice in data["choices"]:
                grp.append(
                    RolloutSample(
                        prompt=prompt,
                        response=choice["text"],
                        finish_reason=choice.get("finish_reason", ""),
                        token_ids=choice.get("logprobs", {}).get("token_ids", []) or [],
                    )
                )
            out.append(grp)
        return RolloutBatch(samples=out)

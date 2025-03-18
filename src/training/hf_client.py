"""Fallback HF `generate` rollout for local dev, CI, and unit tests.

Slower than vLLM by a lot but has no server dependency, which is what we
want in tests. Interface matches VLLMClient.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from src.training.rollout import RolloutBatch, RolloutClient, RolloutSample


@dataclass
class HFConfig:
    device: str = "cuda"
    max_new_tokens: int = 512


class HFClient(RolloutClient):
    def __init__(self, model, tokenizer, cfg: HFConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

    def _generate(self, prompt: str, g: int, **kw) -> list[RolloutSample]:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)
        with torch.inference_mode():
            outs = self.model.generate(
                **inputs,
                max_new_tokens=kw.get("max_new_tokens", self.cfg.max_new_tokens),
                do_sample=True,
                temperature=kw.get("temperature", 1.0),
                top_p=kw.get("top_p", 0.95),
                num_return_sequences=g,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        samples: list[RolloutSample] = []
        for row in outs:
            resp_ids = row[prompt_len:].tolist()
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            samples.append(
                RolloutSample(
                    prompt=prompt, response=text,
                    finish_reason="length", token_ids=resp_ids,
                )
            )
        return samples

    async def sample_group(self, prompt: str, g: int, **kw) -> list[RolloutSample]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate, prompt, g, kw)

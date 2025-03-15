"""Padded token batching + response-only loss masking.

The GRPO loss is only defined on generated tokens, not on the prompt.
This collator packs a mixed-length batch of (prompt_ids, response_ids)
into padded tensors and produces a `response_mask` that the trainer uses
to zero-out the loss on prompt positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class BatchedRollouts:
    input_ids: torch.LongTensor           # [B, T]
    attention_mask: torch.LongTensor      # [B, T]
    response_mask: torch.BoolTensor       # [B, T]  True on response tokens
    prompt_lens: torch.LongTensor         # [B]
    response_lens: torch.LongTensor       # [B]


def collate(
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    pad_id: int,
    max_len: int | None = None,
) -> BatchedRollouts:
    assert len(prompt_ids) == len(response_ids)
    B = len(prompt_ids)

    seqs = [list(p) + list(r) for p, r in zip(prompt_ids, response_ids)]
    plens = [len(p) for p in prompt_ids]
    rlens = [len(r) for r in response_ids]
    T = max(len(s) for s in seqs)
    if max_len is not None:
        T = min(T, max_len)

    ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    resp = torch.zeros((B, T), dtype=torch.bool)

    for i, s in enumerate(seqs):
        L = min(len(s), T)
        ids[i, :L] = torch.tensor(s[:L], dtype=torch.long)
        attn[i, :L] = 1
        # response positions start at plens[i], up to L
        r_start = min(plens[i], L)
        resp[i, r_start:L] = True

    return BatchedRollouts(
        input_ids=ids,
        attention_mask=attn,
        response_mask=resp,
        prompt_lens=torch.tensor(plens, dtype=torch.long),
        response_lens=torch.tensor(rlens, dtype=torch.long),
    )

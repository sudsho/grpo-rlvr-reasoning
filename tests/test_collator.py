"""Collator tests: shapes, masks, boundary cases."""
from __future__ import annotations

import torch

from src.training.collator import collate


def test_shapes_and_response_mask() -> None:
    prompts = [[1, 2, 3], [10, 11]]
    resps = [[100, 101], [200, 201, 202, 203]]
    b = collate(prompts, resps, pad_id=0)

    assert b.input_ids.shape == (2, 6)
    assert (b.attention_mask.sum(dim=1) == torch.tensor([5, 6])).all()
    # response mask should cover exactly the response positions
    assert b.response_mask[0].tolist() == [False, False, False, True, True, False]
    assert b.response_mask[1].tolist() == [False, False, True, True, True, True]


def test_pad_id_fill() -> None:
    prompts = [[1, 2, 3, 4], [5]]
    resps = [[9], [8, 8]]
    b = collate(prompts, resps, pad_id=-1)
    # everything past sequence len is pad
    assert b.input_ids[0, 5] == -1
    assert b.attention_mask[0, 5] == 0


def test_max_len_clip() -> None:
    prompts = [[1] * 20]
    resps = [[2] * 20]
    b = collate(prompts, resps, pad_id=0, max_len=10)
    assert b.input_ids.shape == (1, 10)
    assert b.attention_mask.sum().item() == 10

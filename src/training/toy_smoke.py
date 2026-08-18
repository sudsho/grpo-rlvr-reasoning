"""Tiny-CPU GRPO smoke: one real policy update on a toy causal LM.

The full trainer needs Qwen2.5-Math + vLLM + GPUs. This module proves the
GRPO *plumbing* (reward stack -> group-relative advantages -> response-masked
policy-gradient loss -> optimizer step) on a from-scratch 2-layer toy causal
LM that initializes in milliseconds and trains on CPU with no downloads.

It deliberately reuses the production pieces:
- rewards: `make_composite` + `math_reward` + `math_format_reward`
- advantages: the same group-relative formula as `GRPOWrapper._advantages`
- batching: the real `collate` collator + its response mask

The rollout is mocked (no model.generate): for each prompt we hand-build a
group of candidate responses with deliberately mixed quality so the rewards
have variance and the advantages are non-degenerate. That is exactly what a
real GRPO group looks like, minus the expensive sampling.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.rewards.composite import RewardWeights, make_composite
from src.rewards.format import math_format_reward
from src.rewards.rule_based import math_reward
from src.training.collator import collate

# --------------------------------------------------------------------------- #
# Trivial byte-level tokenizer: no vocab file, no download, deterministic.
# --------------------------------------------------------------------------- #
PAD_ID = 256
VOCAB_SIZE = 257  # 0..255 bytes + one pad id


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8", "replace"))


# --------------------------------------------------------------------------- #
# From-scratch 2-layer toy causal LM.
# --------------------------------------------------------------------------- #
class ToyCausalLM(nn.Module):
    """A minimal GPT-style decoder. Small enough to init + step on CPU fast."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 32,
                 n_head: int = 2, n_layer: int = 2, max_len: int = 512) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=2 * d_model,
                    batch_first=True,
                    dropout=0.0,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        _, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        causal = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_mask=causal)
        return self.lm_head(self.ln_f(x))  # [B, T, V]


# --------------------------------------------------------------------------- #
# Mocked rollout: build a mixed-quality group per prompt (no model.generate).
# --------------------------------------------------------------------------- #
@dataclass
class ToyGroup:
    prompt: str
    responses: list[str]
    gold: str


def mock_rollout_group(prompt: str, gold: str, g: int = 4) -> ToyGroup:
    """Return g candidate responses of deliberately mixed quality.

    Slot 0: correct, well-formatted (reward high).
    Slot 1: correct answer but no <think> tags (loses format credit).
    Slot 2: wrong answer, well-formatted (correctness 0).
    Slot 3+: wrong answer, no format either (reward low).
    """
    wrong = (gold or "0") + "1"  # a guaranteed-wrong answer string
    templates = [
        f"<think>work it through</think>\n\\boxed{{{gold}}}",
        f"the answer is \\boxed{{{gold}}}",
        f"<think>oops</think>\n\\boxed{{{wrong}}}",
        f"just guessing {wrong}",
    ]
    responses = [templates[i % len(templates)] for i in range(g)]
    return ToyGroup(prompt=prompt, responses=responses, gold=gold)


# --------------------------------------------------------------------------- #
# One GRPO-style update.
# --------------------------------------------------------------------------- #
@dataclass
class ToyStepResult:
    loss_before: float
    loss_after: float
    reward_mean: float
    reward_std: float
    n_prompts: int
    n_samples: int
    param_delta: float
    per_group_rewards: list[list[float]]
    per_group_advantages: list[list[float]]


def _group_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantage, matching GRPOWrapper._advantages."""
    mu = statistics.fmean(rewards)
    sd = statistics.pstdev(rewards) if len(rewards) > 1 else 1.0
    sd = max(sd, 1e-6)
    return [(r - mu) / sd for r in rewards]


def _grpo_loss(model: ToyCausalLM, groups: list[ToyGroup],
               advantages: list[list[float]]) -> torch.Tensor:
    """Response-masked policy-gradient loss over all groups.

    loss = - mean_over_samples( advantage * sum_of_response_token_logprobs )
    """
    prompt_ids, response_ids, adv_flat = [], [], []
    for grp, grp_adv in zip(groups, advantages):
        p_ids = encode(grp.prompt)
        for resp, a in zip(grp.responses, grp_adv):
            prompt_ids.append(p_ids)
            response_ids.append(encode(resp))
            adv_flat.append(a)

    batch = collate(prompt_ids, response_ids, pad_id=PAD_ID, max_len=model.max_len)
    logits = model(batch.input_ids)                      # [B, T, V]
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)      # predict pos t from t-1
    tgt = batch.input_ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    resp_mask = batch.response_mask[:, 1:].float()
    seq_logp = (tok_logp * resp_mask).sum(dim=1)         # [B]
    adv = torch.tensor(adv_flat, dtype=seq_logp.dtype)
    return -(adv * seq_logp).mean()


def run_toy_grpo_step(
    prompts_and_gold: list[tuple[str, str]],
    group_size: int = 4,
    lr: float = 1e-3,
    seed: int = 1234,
) -> ToyStepResult:
    """Build groups, score with the real reward stack, take one GRPO step."""
    torch.manual_seed(seed)
    weights = RewardWeights()
    reward_fn = make_composite(math_reward, math_format_reward, weights)

    groups = [mock_rollout_group(p, g, group_size) for p, g in prompts_and_gold]

    per_group_rewards: list[list[float]] = []
    per_group_adv: list[list[float]] = []
    for grp in groups:
        meta = {"gold": grp.gold}
        rewards = [reward_fn(r, meta)["reward"] for r in grp.responses]
        per_group_rewards.append(rewards)
        per_group_adv.append(_group_advantages(rewards))

    model = ToyCausalLM()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    before = _grpo_loss(model, groups, per_group_adv)
    params_before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

    opt.zero_grad()
    before.backward()
    opt.step()

    with torch.no_grad():
        after = _grpo_loss(model, groups, per_group_adv)
        params_after = torch.cat([p.detach().flatten() for p in model.parameters()])
        delta = (params_after - params_before).abs().mean().item()

    flat = [r for row in per_group_rewards for r in row]
    return ToyStepResult(
        loss_before=float(before.item()),
        loss_after=float(after.item()),
        reward_mean=statistics.fmean(flat),
        reward_std=statistics.pstdev(flat) if len(flat) > 1 else 0.0,
        n_prompts=len(groups),
        n_samples=len(flat),
        param_delta=delta,
        per_group_rewards=per_group_rewards,
        per_group_advantages=per_group_adv,
    )

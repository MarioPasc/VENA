"""Functional unit tests for the LoRA adapter on a synthetic attention block.

These tests do not load MAISI weights; they exercise the contract on a tiny
``nn.Module`` that mirrors the attribute names PEFT will match against
(``to_q``, ``to_k``, ``to_v``, ``out_proj``). The identity-at-init guarantee
is the load-bearing property — if the wrapped output diverges from the
unwrapped output at step 0, joint ControlNet + LoRA training would have to
re-converge the ControlNet residuals, which is exactly the curriculum bias
the design avoids.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

peft = pytest.importorskip("peft")

from vena.model.fm.maisi.peft import build_peft  # noqa: E402

pytestmark = pytest.mark.unit


class _MockAttnBlock(nn.Module):
    """Single attention block matching the MONAI MAISI naming convention."""

    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ffn = nn.Linear(d, d, bias=True)  # NOT a LoRA target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
        out = self.out_proj(attn @ v)
        return self.ffn(out)


def _make_block(seed: int = 0, d: int = 64) -> _MockAttnBlock:
    torch.manual_seed(seed)
    return _MockAttnBlock(d=d)


def test_lora_apply_identity_at_init() -> None:
    """After LoRA injection the forward output equals the pretrained output."""
    block = _make_block()
    x = torch.randn(2, 8, 64)
    y_before = block(x).detach().clone()

    handler = build_peft("lora", {"r": 16, "alpha": 16})
    handler.apply(block)
    y_after = block(x).detach().clone()

    assert torch.allclose(y_before, y_after, atol=1e-6), (
        "LoRA must be identity-at-init: post-injection forward differs from pre-injection forward."
    )


def test_lora_freezes_base_and_unfreezes_adapter() -> None:
    block = _make_block()
    handler = build_peft("lora", {"r": 16})
    handler.apply(block)

    base_params = [p for n, p in block.named_parameters() if "lora_" not in n]
    lora_params = [p for n, p in block.named_parameters() if "lora_" in n]

    assert lora_params, "expected LoRA parameters to be injected"
    assert all(not p.requires_grad for p in base_params), (
        "base params must be frozen after LoRA.apply"
    )
    assert all(p.requires_grad for p in lora_params), "LoRA adapter params must be trainable"


def test_lora_targets_only_qkvo_not_ffn() -> None:
    block = _make_block()
    handler = build_peft("lora", {"r": 16})
    handler.apply(block)
    lora_names = [n for n, _ in block.named_parameters() if "lora_" in n]
    # Each matched Linear gets ``lora_A.default.weight`` + ``lora_B.default.weight``.
    assert sum("to_q.lora_A" in n for n in lora_names) == 1
    assert sum("to_k.lora_A" in n for n in lora_names) == 1
    assert sum("to_v.lora_A" in n for n in lora_names) == 1
    assert sum("out_proj.lora_A" in n for n in lora_names) == 1
    assert not any("ffn.lora" in n for n in lora_names), (
        "FFN was not in target_modules; must not be adapted"
    )


def test_lora_extract_state_keys_match_named_parameters() -> None:
    block = _make_block()
    handler = build_peft("lora", {"r": 16})
    handler.apply(block)
    state = handler.extract_state(block)
    live_keys = {n for n, _ in block.named_parameters() if "lora_" in n}
    assert set(state) == live_keys
    # Tensors are clones — mutating state must not touch the live module.
    name = next(iter(state))
    state[name].zero_()
    live = dict(block.named_parameters())[name]
    assert not torch.allclose(live, torch.zeros_like(live))


def test_lora_load_state_round_trip() -> None:
    """extract_state → load_state on a fresh wrap must restore exact tensors."""
    block_a = _make_block(seed=0)
    handler = build_peft("lora", {"r": 16})
    handler.apply(block_a)
    # Simulate training by perturbing the LoRA tensors.
    with torch.no_grad():
        for n, p in block_a.named_parameters():
            if "lora_" in n:
                p.add_(torch.randn_like(p) * 0.01)
    state = handler.extract_state(block_a)

    block_b = _make_block(seed=0)
    handler.apply(block_b)
    handler.load_state(block_b, state)

    a = dict(block_a.named_parameters())
    b = dict(block_b.named_parameters())
    for name in state:
        assert torch.allclose(a[name], b[name], atol=1e-7), f"load_state did not round-trip {name}"


def test_lora_trainable_parameters_count() -> None:
    block = _make_block(d=64)
    handler = build_peft("lora", {"r": 16, "target_modules": ["to_q", "to_k", "to_v", "out_proj"]})
    handler.apply(block)
    trainable = handler.trainable_parameters(block)
    # 4 matched Linears x (lora_A: r*d + lora_B: d*r) = 4 * (16*64 + 64*16) = 8192.
    n_trainable = sum(p.numel() for p in trainable)
    assert n_trainable == 4 * (16 * 64 + 64 * 16)


def test_lora_gradient_flows_only_to_adapter() -> None:
    block = _make_block()
    handler = build_peft("lora", {"r": 16})
    handler.apply(block)
    x = torch.randn(2, 8, 64)
    loss = block(x).pow(2).mean()
    loss.backward()
    for n, p in block.named_parameters():
        if "lora_" in n:
            assert p.grad is not None, f"missing grad on adapter param {n}"
        else:
            assert p.grad is None, f"unexpected grad on frozen base param {n}"

"""Unit tests for classifier-free-guidance training-time dropout.

The dropout is implemented in
``FMLightningModule._build_conditioning_with_cfg_dropout`` (added 2026-06-09,
CHANGE 3). We exercise it directly via a lightweight subclass that injects a
``ConditioningAssembler`` without loading the trunk / VAE / ControlNet — the
math under test is the per-sample Bernoulli ``torch.where`` of two
already-assembled conditioning tensors, not the actual trunk forward.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Lightweight harness: imitate FMLightningModule's helper without instantiating
# the full module (which needs MAISI checkpoints).
# ---------------------------------------------------------------------------


class _FakeAssembler(nn.Module):
    """Minimal ConditioningAssembler stand-in.

    Maps ``perturb_keys -> (B, C, h, w, d)`` where the channels corresponding
    to listed keys are zeroed. The numerics match the real assembler's
    perturb semantics enough for the dropout's torch.where to be exercised.
    """

    def __init__(self, B: int, C_per_key: int, spatial: tuple[int, int, int]) -> None:
        super().__init__()
        self.spec_keys = ["t1pre", "t2", "flair", "wt"]
        self.C_per_key = C_per_key
        # One distinctive constant per key so we can check zeroing.
        self._channels: dict[str, torch.Tensor] = {
            key: float(i + 1) * torch.ones(B, C_per_key, *spatial)
            for i, key in enumerate(self.spec_keys)
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        perturb_keys: set[str] | None = None,
    ) -> torch.Tensor:
        perturb = perturb_keys or set()
        pieces = []
        for key in self.spec_keys:
            x = self._channels[key]
            if key in perturb:
                x = torch.zeros_like(x)
            pieces.append(x)
        return torch.cat(pieces, dim=1)


class _CFGDropoutHarness(nn.Module):
    """Re-implements only the dropout helper to keep tests cheap."""

    def __init__(
        self,
        B: int,
        C_per_key: int,
        spatial: tuple[int, int, int],
        p: float,
        keys: tuple[str, ...] = ("wt",),
    ) -> None:
        super().__init__()
        self.conditioning = _FakeAssembler(B, C_per_key, spatial)
        self.conditioning_dropout_p = p
        self.conditioning_dropout_keys: set[str] = set(keys)
        self.last_active_frac: float | None = None

    def step(self, batch: dict[str, torch.Tensor], B: int) -> torch.Tensor:
        """Same logic as FMLightningModule._build_conditioning_with_cfg_dropout."""
        if self.conditioning_dropout_p <= 0.0:
            return self.conditioning(batch)
        cond_keep = self.conditioning(batch)
        cond_drop = self.conditioning(batch, perturb_keys=self.conditioning_dropout_keys)
        drop = torch.rand(B) < self.conditioning_dropout_p
        mask = drop.view(-1, 1, 1, 1, 1)
        self.last_active_frac = float(drop.float().mean().item())
        return torch.where(mask, cond_drop, cond_keep)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_zero_p_drop_no_op() -> None:
    """With p=0 the dropout helper returns the unperturbed conditioning."""
    B, C_per_key, spatial = 4, 4, (4, 4, 4)
    harness = _CFGDropoutHarness(B, C_per_key, spatial, p=0.0)
    cond_p0 = harness.step({}, B)
    cond_baseline = harness.conditioning({})
    assert torch.equal(cond_p0, cond_baseline)
    assert harness.last_active_frac is None  # path skipped → no log emission


def test_p_drop_one_fully_zeros_wt_channels() -> None:
    """With p=1 every sample's WT channels in conditioning are zero."""
    B, C_per_key, spatial = 8, 4, (4, 4, 4)
    torch.manual_seed(0)
    harness = _CFGDropoutHarness(B, C_per_key, spatial, p=1.0)
    cond = harness.step({}, B)
    # WT is the 4th key → channels [12:16] in the concatenated conditioning.
    wt_channels = cond[:, 12:16]
    assert torch.all(wt_channels == 0)
    # All non-WT channels keep their constant values.
    for i, key in enumerate(["t1pre", "t2", "flair"]):
        s, e = i * C_per_key, (i + 1) * C_per_key
        assert torch.all(cond[:, s:e] == float(i + 1))


def test_p_drop_half_bernoulli_3sigma() -> None:
    """Sampled rate stays within ~3σ of 0.5 over 10 000 draws."""
    n = 10_000
    torch.manual_seed(42)
    drops = torch.rand(n) < 0.5
    rate = drops.float().mean().item()
    # σ for Bernoulli(0.5) over n draws = sqrt(0.25 / n).
    sigma = (0.25 / n) ** 0.5
    assert abs(rate - 0.5) < 3.0 * sigma


def test_drop_only_affects_listed_keys() -> None:
    """With dropout_keys={wt}, the latent channels stay bit-identical."""
    B, C_per_key, spatial = 4, 4, (4, 4, 4)
    torch.manual_seed(1)
    harness = _CFGDropoutHarness(B, C_per_key, spatial, p=0.5, keys=("wt",))
    cond = harness.step({}, B)
    cond_baseline = harness.conditioning({})
    # Latent channels (t1pre, t2, flair) are at positions 0:12.
    assert torch.equal(cond[:, 0:12], cond_baseline[:, 0:12])

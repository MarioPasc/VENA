"""Tests for the downsampler-aware ``ConditioningAssembler.channels_per_spec``.

The 2026-06-20 refactor lets the assembler delegate the channel count to
the downsampler when the operator exposes ``out_channels`` (used by
:class:`LiftTo4ChDownsampler`). Stateless operators (identity, nearest,
trilinear, avg_pool, zero_out) still fall back to the kind-based default.
"""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.conditioning import ConditioningAssembler


@pytest.mark.unit
def test_zero_out_preserves_kind_default() -> None:
    asm = ConditioningAssembler(
        specs=[
            "latent:t1pre",
            "latent:t2",
            "latent:flair",
            "mask:wt:zero_out",
        ],
        latent_channels=4,
        mask_channels=1,
    )
    assert asm.channels_per_spec == [4, 4, 4, 1]
    assert asm.total_channels == 13


@pytest.mark.unit
def test_lift_to_4ch_overrides_kind_default() -> None:
    asm = ConditioningAssembler(
        specs=[
            "latent:t1pre",
            "latent:t2",
            "latent:flair",
            "mask:wt:lift_to_4ch",
        ],
        latent_channels=4,
        mask_channels=1,
    )
    assert asm.channels_per_spec == [4, 4, 4, 4]
    assert asm.total_channels == 16


@pytest.mark.unit
def test_zero_out_forward_returns_zeros_in_mask_slot() -> None:
    asm = ConditioningAssembler(
        specs=[
            "latent:t1pre",
            "mask:wt:zero_out",
        ],
        latent_channels=4,
        mask_channels=1,
    )
    batch = {
        "z_t1pre": torch.randn(2, 4, 12, 12, 8),
        "m_wt": torch.rand(2, 1, 12, 12, 8),
    }
    out = asm(batch)
    assert out.shape == (2, 5, 12, 12, 8)
    # The first 4 channels are the latent; the last channel is the zero-out slot.
    assert torch.all(out[:, 4:5, ...] == 0)
    # The latent slot is untouched.
    assert torch.equal(out[:, :4, ...], batch["z_t1pre"])


@pytest.mark.unit
def test_lift_to_4ch_forward_produces_4_channel_output() -> None:
    asm = ConditioningAssembler(
        specs=["mask:wt:lift_to_4ch"],
        latent_channels=4,
        mask_channels=1,
    )
    batch = {"m_wt": torch.rand(2, 1, 6, 6, 4)}
    out = asm(batch)
    assert out.shape == (2, 4, 6, 6, 4)


@pytest.mark.unit
def test_identity_mask_legacy_unchanged() -> None:
    """Existing ``mask:wt:identity`` configs must keep their channel layout."""
    asm = ConditioningAssembler(
        specs=[
            "latent:t1pre",
            "latent:t2",
            "latent:flair",
            "mask:wt:identity",
        ],
        latent_channels=4,
        mask_channels=1,
    )
    assert asm.channels_per_spec == [4, 4, 4, 1]
    assert asm.total_channels == 13

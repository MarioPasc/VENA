"""Unit tests for the MAISI preprocessing helpers."""

from __future__ import annotations

import pytest
import torch

from vena.model.autoencoder.maisi.exceptions import ShapeContractError
from vena.model.autoencoder.maisi.preprocessing import (
    crop_to_original,
    pad_depth_to_multiple_of,
    percentile_normalise,
)


@pytest.mark.unit
def test_percentile_normalise_maps_to_unit_range() -> None:
    rng = torch.Generator().manual_seed(0)
    x = torch.randn((2, 1, 8, 8, 8), generator=rng)
    y = percentile_normalise(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert y.min().item() >= 0.0 - 1e-6
    assert y.max().item() <= 1.0 + 1e-6


@pytest.mark.unit
def test_percentile_normalise_constant_volume_is_stable() -> None:
    x = torch.full((1, 1, 4, 4, 4), 3.14)
    y = percentile_normalise(x)
    assert torch.isfinite(y).all()
    # All-constant input → division by eps; result lives in [0, 1] but is not NaN.
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


@pytest.mark.unit
def test_pad_depth_already_multiple_is_noop() -> None:
    x = torch.zeros((1, 1, 4, 4, 16))
    y, pad = pad_depth_to_multiple_of(x, base=8)
    assert y.shape == x.shape
    assert pad.after == 0
    assert pad.padded_depth == 16


@pytest.mark.unit
def test_pad_then_crop_roundtrips() -> None:
    rng = torch.Generator().manual_seed(0)
    x = torch.randn((2, 3, 8, 8, 13), generator=rng)
    y, pad = pad_depth_to_multiple_of(x, base=8)
    assert y.shape[-1] == 16
    assert pad.after == 3
    back = crop_to_original(y, pad)
    assert back.shape == x.shape
    assert torch.allclose(back, x)


@pytest.mark.unit
def test_pad_rejects_wrong_rank() -> None:
    with pytest.raises(ShapeContractError):
        pad_depth_to_multiple_of(torch.zeros((1, 8, 8, 8)))


@pytest.mark.unit
def test_percentile_rejects_wrong_rank() -> None:
    with pytest.raises(ShapeContractError):
        percentile_normalise(torch.zeros((1, 8, 8, 8)))


@pytest.mark.unit
def test_percentile_normalise_mask_overrides_foreground_heuristic() -> None:
    """Brain-mask path must include negative intra-brain voxels, mirroring
    BraTS-Africa z-score data — see ``.claude/notes/data/2026-06-18_data_audit.md``."""
    rng = torch.Generator().manual_seed(7)
    x = torch.randn((1, 1, 8, 8, 8), generator=rng)
    # Half of foreground is negative — simulating z-score brain tissue.
    mask = torch.zeros_like(x)
    mask[..., :8, :8, :4] = 1.0  # brain occupies left half
    y_mask = percentile_normalise(x, mask=mask)
    y_fg = percentile_normalise(x, foreground_only=True, foreground_threshold=0.0)
    # The two paths disagree because the mask path includes negative voxels.
    assert not torch.allclose(y_mask, y_fg)
    # Out-of-mask voxels still go through clip(0,1).
    assert y_mask.min().item() >= 0.0 - 1e-6
    assert y_mask.max().item() <= 1.0 + 1e-6


@pytest.mark.unit
def test_percentile_normalise_mask_shape_validation() -> None:
    x = torch.zeros((1, 1, 8, 8, 8))
    bad = torch.zeros((2, 1, 8, 8, 8))  # batch mismatch
    with pytest.raises(ShapeContractError):
        percentile_normalise(x, mask=bad)
    bad_spatial = torch.zeros((1, 1, 4, 4, 4))
    with pytest.raises(ShapeContractError):
        percentile_normalise(x, mask=bad_spatial)


@pytest.mark.unit
def test_percentile_normalise_default_unchanged_when_no_mask() -> None:
    """Existing call sites without a mask must produce byte-identical output."""
    rng = torch.Generator().manual_seed(11)
    x = torch.randn((2, 1, 8, 8, 8), generator=rng)
    y_a = percentile_normalise(x, foreground_only=True)
    y_b = percentile_normalise(x, foreground_only=True, mask=None)
    assert torch.equal(y_a, y_b)

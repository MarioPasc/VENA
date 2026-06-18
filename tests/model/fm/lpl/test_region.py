"""Unit tests for :mod:`vena.model.fm.lpl.region`.

Critical pieces:

* The §4.7b NN-upsample sanity check: a 1-voxel WT at latent index
  ``(0, 0, 0)`` upsamples to the 2×2×2 corner of the 2× grid, and the
  same WT at ``(H-1, W-1, D-1)`` maps to the opposite corner — *not*
  one off-by-one voxel.
* Soft variant boundary smoothness — trilinear upsample produces a
  monotone ramp at a step edge.
* :func:`region_weight_map` linearity (binary mode) + out-of-brain
  voxels zeroed.
* :func:`soft_wt_from_tumor_latent` clip+sum behaviour and the optional
  ``brain_lat`` masking.
"""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.lpl import (
    region_weight_map,
    resample_region_to_block,
    soft_wt_from_tumor_latent,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# soft_wt_from_tumor_latent
# ---------------------------------------------------------------------------


def test_soft_wt_clip_sum() -> None:
    """Channel sum is clipped to ``[0, 1]`` (channels can over-saturate)."""
    tumor = torch.full((1, 3, 2, 2, 2), 0.6)  # sum = 1.8 → clipped to 1.0
    soft = soft_wt_from_tumor_latent(tumor)
    assert soft.shape == (1, 1, 2, 2, 2)
    assert (soft == 1.0).all()


def test_soft_wt_masked_by_brain() -> None:
    tumor = torch.full((1, 3, 2, 2, 2), 0.6)
    brain = torch.zeros(1, 1, 2, 2, 2)
    brain[..., 0, 0, 0] = 1.0
    soft = soft_wt_from_tumor_latent(tumor, brain_lat=brain)
    assert soft[..., 0, 0, 0].item() == 1.0
    # All other voxels zeroed by the brain mask.
    other = soft.clone()
    other[..., 0, 0, 0] = 0.0
    assert (other == 0.0).all()


def test_soft_wt_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match=r"\(B, 3, H, W, D\)"):
        soft_wt_from_tumor_latent(torch.zeros(1, 2, 2, 2, 2))


# ---------------------------------------------------------------------------
# resample_region_to_block — the load-bearing §4.7b sanity check
# ---------------------------------------------------------------------------


def test_nn_upsample_preserves_corner_position() -> None:
    """A 1-voxel WT at latent corner (0,0,0) maps to the 2×2×2 corner at
    2× target shape (§4.7b sanity)."""
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[0, 0, 0, 0, 0] = 1.0
    up = resample_region_to_block(mask, target_shape=(8, 8, 8), mode="nearest")
    # The first 2 voxels along every axis should be 1.
    assert (up[0, 0, :2, :2, :2] == 1.0).all()
    # Everything else should be zero.
    rest = up.clone()
    rest[0, 0, :2, :2, :2] = 0.0
    assert (rest == 0.0).all()


def test_nn_upsample_preserves_opposite_corner() -> None:
    """Mirror: WT at the opposite latent corner maps to the opposite 2× corner."""
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[0, 0, 3, 3, 3] = 1.0
    up = resample_region_to_block(mask, target_shape=(8, 8, 8), mode="nearest")
    assert (up[0, 0, 6:, 6:, 6:] == 1.0).all()
    rest = up.clone()
    rest[0, 0, 6:, 6:, 6:] = 0.0
    assert (rest == 0.0).all()


def test_nop_resample_when_shape_matches() -> None:
    """Same-shape input must come back identical (no needless interpolate)."""
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 1, 2, 2, 1)
    out = resample_region_to_block(mask, target_shape=(2, 2, 1), mode="nearest")
    assert torch.equal(out, mask)


def test_trilinear_smooths_boundary() -> None:
    """Trilinear upsample produces values strictly between 0 and 1 at the
    boundary of a step-edge mask (NN would not)."""
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[0, 0, :2] = 1.0  # half-volume step
    up = resample_region_to_block(mask, target_shape=(8, 8, 8), mode="trilinear")
    # Some interior voxels should be in (0, 1).
    fractional = ((up > 0.0) & (up < 1.0)).any().item()
    assert fractional, "trilinear should have produced fractional boundary values"


def test_rejects_wrong_dims() -> None:
    with pytest.raises(ValueError, match=r"\(B, 1, H, W, D\)"):
        resample_region_to_block(torch.zeros(1, 2, 4, 4, 4), target_shape=(8, 8, 8))


# ---------------------------------------------------------------------------
# region_weight_map
# ---------------------------------------------------------------------------


def test_binary_region_weight_split() -> None:
    """Inside brain: voxel ∈ WT → alpha_wt; voxel ∉ WT → alpha_notwt."""
    m_wt = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).view(1, 1, 2, 2, 1)
    m_brain = torch.tensor([[1.0, 1.0, 1.0, 0.0]]).view(1, 1, 2, 2, 1)
    w = region_weight_map(m_wt, m_brain, alpha_wt=2.0, alpha_notwt=3.0, soft=False)
    # voxel (0,0): not WT, in brain → 3
    # voxel (0,1): WT, in brain → 2
    # voxel (1,0): not WT, in brain → 3
    # voxel (1,1): not WT, NOT in brain → 0 (out-of-brain zeroed)
    expected = torch.tensor([3.0, 2.0, 3.0, 0.0]).view(1, 1, 2, 2, 1)
    assert torch.equal(w, expected)


def test_soft_region_weight_linearity() -> None:
    """In soft mode, ``w = m_brain * (a_wt * m_wt + a_notwt * (1 - m_wt))``."""
    m_wt = torch.tensor([0.0, 0.5, 1.0, 0.25]).view(1, 1, 2, 2, 1)
    m_brain = torch.ones_like(m_wt)
    w = region_weight_map(m_wt, m_brain, alpha_wt=2.0, alpha_notwt=4.0, soft=True)
    expected = torch.tensor([4.0, 3.0, 2.0, 3.5]).view(1, 1, 2, 2, 1)
    assert torch.allclose(w, expected)


def test_out_of_brain_voxels_zeroed_both_modes() -> None:
    m_wt = torch.tensor([0.7, 0.2]).view(1, 1, 1, 2, 1)
    m_brain = torch.tensor([0.0, 1.0]).view(1, 1, 1, 2, 1)
    w_binary = region_weight_map(m_wt, m_brain, alpha_wt=1.0, alpha_notwt=1.0, soft=False)
    w_soft = region_weight_map(m_wt, m_brain, alpha_wt=1.0, alpha_notwt=1.0, soft=True)
    assert w_binary[..., 0, 0, 0].item() == 0.0
    assert w_soft[..., 0, 0, 0].item() == 0.0


def test_region_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        region_weight_map(
            torch.zeros(1, 1, 2, 2, 1),
            torch.zeros(1, 1, 2, 3, 1),
            alpha_wt=1.0,
            alpha_notwt=1.0,
        )

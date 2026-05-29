"""Crop-box geometry contracts: brain containment + crop/pad round-trip.

The multi-cohort corpus stores native volumes plus a per-scan crop origin; the
encoder crops/pads each cohort onto a common box so latents share a grid. These
tests lock the two halves of that contract: the data-layer origin computation
(``compute_crop_origin``) and the model-layer tensor ops (``apply_crop_pad`` /
``invert_crop_pad``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vena.data.h5.shared.crop import CropGeometryError, compute_crop_origin
from vena.model.autoencoder.maisi.preprocessing import (
    CropPadSpec,
    apply_crop_pad,
    invert_crop_pad,
)

BOX = (192, 224, 192)


def _brain_mask(shape, bbox):
    m = np.zeros(shape, dtype=np.uint8)
    (a0, a1), (b0, b1), (c0, c1) = bbox
    m[a0:a1, b0:b1, c0:c1] = 1
    return m


@pytest.mark.unit
def test_ucsf_like_crop_contains_brain() -> None:
    """A 240x240x155 (UCSF-like) volume: in-plane crop + depth pad, brain inside."""
    mask = _brain_mask((240, 240, 155), ((46, 195), (30, 224), (5, 155)))
    origin = compute_crop_origin(mask, BOX)
    lo = np.argwhere(mask).min(0)
    hi = np.argwhere(mask).max(0)
    for i in range(3):
        assert origin[i] <= lo[i]
        assert hi[i] < origin[i] + BOX[i]


@pytest.mark.unit
def test_brats_like_pure_pad() -> None:
    """A 182x218x182 (BraTS-like) volume is smaller than the box on every axis."""
    mask = _brain_mask((182, 218, 182), ((16, 166), (12, 202), (14, 168)))
    origin = compute_crop_origin(mask, BOX)
    # All axes pad (box >= native) → origin negative or zero, brain contained.
    lo = np.argwhere(mask).min(0)
    hi = np.argwhere(mask).max(0)
    for i in range(3):
        assert origin[i] <= lo[i]
        assert hi[i] < origin[i] + BOX[i]


@pytest.mark.unit
def test_box_too_small_raises() -> None:
    """A box smaller than the brain extent on some axis must fail loudly."""
    mask = _brain_mask((240, 240, 200), ((10, 230), (10, 230), (10, 190)))  # R-L extent 220 > 192
    with pytest.raises(CropGeometryError):
        compute_crop_origin(mask, BOX)


@pytest.mark.unit
def test_empty_mask_centres_box() -> None:
    """An empty mask falls back to the geometric centre (never crashes)."""
    mask = np.zeros((240, 240, 155), dtype=np.uint8)
    origin = compute_crop_origin(mask, BOX)
    assert origin == (round(240 / 2 - 192 / 2), round(240 / 2 - 224 / 2), round(155 / 2 - 192 / 2))


@pytest.mark.unit
@pytest.mark.parametrize(
    "native,origin",
    [((240, 240, 155), (24, 8, -18)), ((182, 218, 182), (-5, -3, -5))],
)
def test_apply_crop_pad_shape_and_inverse(native, origin) -> None:
    """apply_crop_pad yields the box; invert restores native; overlap is identity."""
    spec = CropPadSpec(crop_origin=origin, native_shape=native, target_shape=BOX)
    x = torch.randn(1, 1, *native)
    y = apply_crop_pad(x, spec)
    assert tuple(y.shape) == (1, 1, *BOX)
    back = invert_crop_pad(y, spec)
    assert tuple(back.shape) == (1, 1, *native)
    # The overlap region (inside both native and box) round-trips exactly.
    s = [(max(0, origin[i]), min(native[i], origin[i] + BOX[i])) for i in range(3)]
    ov_x = x[:, :, s[0][0]:s[0][1], s[1][0]:s[1][1], s[2][0]:s[2][1]]
    ov_b = back[:, :, s[0][0]:s[0][1], s[1][0]:s[1][1], s[2][0]:s[2][1]]
    assert torch.equal(ov_x, ov_b)


@pytest.mark.unit
def test_apply_crop_pad_rejects_wrong_native() -> None:
    spec = CropPadSpec(crop_origin=(0, 0, 0), native_shape=(240, 240, 155), target_shape=BOX)
    with pytest.raises(Exception):
        apply_crop_pad(torch.randn(1, 1, 100, 100, 100), spec)

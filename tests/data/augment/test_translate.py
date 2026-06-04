"""Unit tests for :class:`vena.data.augment.transforms.translate.Translate`."""

from __future__ import annotations

import random

import pytest
import torch

from vena.data.augment.online.base import LatentAugmentationError
from vena.data.augment.online.transforms.translate import Translate


@pytest.fixture()
def sample_batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "patient_id": "TEST-002",
        "z_t1pre": torch.randn(4, 8, 8, 8, generator=g),
        "z_t2": torch.randn(4, 8, 8, 8, generator=g),
        "z_flair": torch.randn(4, 8, 8, 8, generator=g),
        "z_t1c": torch.randn(4, 8, 8, 8, generator=g),
        "m_wt": torch.randn(1, 8, 8, 8, generator=g),
    }


def test_max_voxels_must_be_multiple_of_four() -> None:
    with pytest.raises(LatentAugmentationError):
        Translate(p=1.0, max_voxels=5)


def test_negative_max_voxels_rejected() -> None:
    with pytest.raises(LatentAugmentationError):
        Translate(p=1.0, max_voxels=-4)


def test_unknown_axis_rejected() -> None:
    with pytest.raises(LatentAugmentationError):
        Translate(p=1.0, max_voxels=4, axes=("x",))


def test_zero_shift_is_identity(sample_batch: dict[str, torch.Tensor]) -> None:
    aug = Translate(p=1.0, max_voxels=4)
    out = aug.apply_latent(dict(sample_batch), {"shifts_img": {"h": 0, "w": 0, "d": 0}})
    for k in ("z_t1pre", "z_t1c", "m_wt"):
        torch.testing.assert_close(out[k], sample_batch[k])


def test_latent_shift_is_image_shift_divided_by_compression() -> None:
    """Image shift of 8 voxels ↔ latent shift of 2 voxels."""
    aug = Translate(p=1.0, max_voxels=8)
    batch = {"z_t1c": torch.arange(4 * 8 * 8 * 8).reshape(4, 8, 8, 8).float()}
    out = aug.apply_latent(dict(batch), {"shifts_img": {"h": 8, "w": 0, "d": 0}})
    expected_zero_rows = 2  # 8 voxels / 4 compression
    assert torch.all(out["z_t1c"][:, :expected_zero_rows, :, :] == 0.0)


def test_apply_image_preserves_shape() -> None:
    x = torch.randn(24, 24, 16)
    out = Translate(p=1.0, max_voxels=8).apply_image(x, {"shifts_img": {"h": 4, "w": -4, "d": 0}})
    assert out.shape == x.shape


def test_sample_params_only_uses_rng() -> None:
    aug = Translate(p=1.0, max_voxels=8, axes=("h", "w"))
    rng_a = random.Random(123)
    rng_b = random.Random(123)
    assert aug.sample_params(rng_a) == aug.sample_params(rng_b)


def test_param_tag_skips_zero_axes() -> None:
    aug = Translate(p=1.0, max_voxels=8)
    tag = aug.param_tag({"shifts_img": {"h": 4, "w": 0, "d": -4}})
    # Sign-encoded as p/n so the tag is safe inside the "+"-joined combo str.
    assert "hp4" in tag and "dn4" in tag
    assert "w" not in tag and "+" not in tag
    assert tag == "translate_hp4dn4"

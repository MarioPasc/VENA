"""Unit tests for the rotation operators."""

from __future__ import annotations

import random

import pytest
import torch

from vena.data.augment.online.base import LatentAugmentationError
from vena.data.augment.online.transforms.rotate import RotateRoll, RotateYaw


@pytest.fixture()
def sample_batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "z_t1pre": torch.randn(4, 16, 16, 12, generator=g),
        "z_t1c": torch.randn(4, 16, 16, 12, generator=g),
        "m_wt": torch.randn(1, 16, 16, 12, generator=g),
    }


def test_negative_max_deg_rejected() -> None:
    with pytest.raises(LatentAugmentationError):
        RotateYaw(p=1.0, max_deg=-1.0)
    with pytest.raises(LatentAugmentationError):
        RotateRoll(p=1.0, max_deg=0.0)


def test_zero_angle_is_identity(sample_batch: dict[str, torch.Tensor]) -> None:
    aug = RotateYaw(p=1.0, max_deg=5.0)
    out = aug.apply_latent(dict(sample_batch), {"deg": 0.0})
    for k in ("z_t1c", "m_wt"):
        torch.testing.assert_close(out[k], sample_batch[k])


def test_shape_preserved_under_rotation(sample_batch: dict[str, torch.Tensor]) -> None:
    aug = RotateRoll(p=1.0, max_deg=5.0)
    out = aug.apply_latent(dict(sample_batch), {"deg": 5.0})
    for k in ("z_t1pre", "z_t1c", "m_wt"):
        assert out[k].shape == sample_batch[k].shape


def test_apply_image_returns_3d() -> None:
    x = torch.randn(24, 24, 16)
    out = RotateYaw(p=1.0, max_deg=5.0).apply_image(x, {"deg": 2.0})
    assert out.shape == x.shape
    assert out.ndim == 3


def test_sample_params_in_range() -> None:
    aug = RotateRoll(p=1.0, max_deg=5.0)
    rng = random.Random(0)
    for _ in range(100):
        params = aug.sample_params(rng)
        assert -5.0 <= params["deg"] <= 5.0


def test_param_tag_buckets_to_integer_deg() -> None:
    # Sign is encoded with "p"/"n" so the tag has no "+" character that would
    # collide with the pipeline's "+"-separated combination string.
    assert RotateYaw(p=1.0).param_tag({"deg": 2.4}) == "rotate_yaw_p2"
    assert RotateYaw(p=1.0).param_tag({"deg": -4.6}) == "rotate_yaw_n5"

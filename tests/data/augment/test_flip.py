"""Unit tests for :class:`vena.data.augment.transforms.flip.FlipLR`."""

from __future__ import annotations

import random

import pytest
import torch

from vena.data.augment.online.transforms.flip import FlipLR


@pytest.fixture()
def sample_batch() -> dict[str, torch.Tensor]:
    """One-sample latent dict shaped like ``LatentH5Dataset._read_one``."""
    g = torch.Generator().manual_seed(0)
    return {
        "patient_id": "TEST-001",
        "z_t1pre": torch.randn(4, 6, 8, 5, generator=g),
        "z_t2": torch.randn(4, 6, 8, 5, generator=g),
        "z_flair": torch.randn(4, 6, 8, 5, generator=g),
        "z_t1c": torch.randn(4, 6, 8, 5, generator=g),
        "m_wt": (torch.rand(1, 6, 8, 5, generator=g) > 0.5).float(),
    }


def test_flip_lr_idempotent(sample_batch: dict[str, torch.Tensor]) -> None:
    aug = FlipLR(p=1.0)
    rng = random.Random(0)
    params = aug.sample_params(rng)
    original = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}
    once = aug.apply_latent(dict(sample_batch), params)
    twice = aug.apply_latent(once, params)
    for k in ("z_t1pre", "z_t2", "z_flair", "z_t1c", "m_wt"):
        torch.testing.assert_close(twice[k], original[k])


def test_flip_lr_all_keys_flipped_together(
    sample_batch: dict[str, torch.Tensor],
) -> None:
    aug = FlipLR(p=1.0)
    flipped = aug.apply_latent(dict(sample_batch), {})
    for k in ("z_t1pre", "z_t2", "z_flair", "z_t1c", "m_wt"):
        torch.testing.assert_close(flipped[k], torch.flip(sample_batch[k], dims=[-3]))


def test_flip_lr_apply_image_shape() -> None:
    x = torch.randn(24, 32, 20)
    out = FlipLR(p=1.0).apply_image(x, {})
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.flip(x, dims=[-3]))


def test_flip_lr_apply_image_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError):
        FlipLR(p=1.0).apply_image(torch.randn(4, 24, 32, 20), {})


def test_flip_lr_param_tag() -> None:
    assert FlipLR(p=1.0).param_tag({}) == "flip_lr"


def test_flip_lr_invalid_probability() -> None:
    with pytest.raises(Exception):  # noqa: B017 — concrete class is LatentAugmentationError
        FlipLR(p=1.5)

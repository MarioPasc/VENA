"""Unit tests for the gamma operator."""

from __future__ import annotations

import pytest
import torch

from vena.data.augment.online.base import LatentAugmentationError
from vena.data.augment.online.transforms.gamma import Gamma


def test_invalid_range_rejected() -> None:
    with pytest.raises(LatentAugmentationError):
        Gamma(p=1.0, gamma_min=1.2, gamma_max=0.8)
    with pytest.raises(LatentAugmentationError):
        Gamma(p=1.0, gamma_min=0.0)


def test_apply_image_stays_in_0_1() -> None:
    x = torch.rand(24, 24, 16)
    out = Gamma(p=1.0).apply_image(x, {"gamma": 1.5})
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_apply_latent_scales_only_modalities() -> None:
    g = torch.Generator().manual_seed(0)
    batch = {
        "z_t1c": torch.randn(4, 8, 8, 8, generator=g),
        "m_wt": torch.randn(1, 8, 8, 8, generator=g),
    }
    mask_before = batch["m_wt"].clone()
    out = Gamma(p=1.0).apply_latent(dict(batch), {"gamma": 1.5})
    torch.testing.assert_close(out["z_t1c"], batch["z_t1c"] * 1.5)
    # Mask must be untouched — gamma has no geometric effect.
    torch.testing.assert_close(out["m_wt"], mask_before)


def test_param_tag_one_decimal() -> None:
    assert Gamma(p=1.0).param_tag({"gamma": 0.83}) == "gamma_0.8"
    assert Gamma(p=1.0).param_tag({"gamma": 1.25}) == "gamma_1.2"

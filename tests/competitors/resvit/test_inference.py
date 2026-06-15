"""Unit tests for ResViT inference helpers — pure math, no checkpoint load.

The full inference path (model import + sampling) is GPU-only; tests for it
live elsewhere with the ``gpu`` / ``slow`` markers. Here we cover the small
pure helpers (PSNR / SSIM / crop) so a regression in arithmetic is caught
in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vena.competitors.resvit.inference import _crop_to, _psnr, _ssim_3d

pytestmark = pytest.mark.unit


def test_psnr_identity_is_infinite() -> None:
    x = np.random.default_rng(0).uniform(0, 1, size=(10, 10, 10)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    assert _psnr(x, x, mask) == float("inf")


def test_psnr_finite_for_perturbed() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(10, 10, 10)).astype(np.float32)
    y = x + 0.1
    mask = np.ones_like(x, dtype=bool)
    psnr = _psnr(y, x, mask)
    assert np.isfinite(psnr)
    # 10*log10(1 / 0.01) = 20.0
    np.testing.assert_allclose(psnr, 20.0, rtol=1e-4)


def test_psnr_nan_when_mask_empty() -> None:
    x = np.ones((5, 5, 5), dtype=np.float32)
    mask = np.zeros_like(x, dtype=bool)
    assert np.isnan(_psnr(x, x, mask))


def test_ssim_identity_is_one() -> None:
    x = np.random.default_rng(0).uniform(0, 1, size=(20, 20, 20)).astype(np.float32)
    np.testing.assert_allclose(_ssim_3d(x, x), 1.0, rtol=1e-3)


def test_crop_to_centred() -> None:
    x = torch.arange(256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256)
    y = _crop_to(x, 240, 240)
    assert y.shape == (1, 1, 240, 240)
    torch.testing.assert_close(y[0, 0], x[0, 0, 8:248, 8:248])

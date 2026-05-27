"""Unit tests for LatentMetrics + ImageMetrics."""

from __future__ import annotations

import math

import pytest
import torch

from vena.model.fm.metrics import ImageMetrics, LatentMetrics


@pytest.mark.unit
def test_latent_mse_zero_on_identical_inputs() -> None:
    z = torch.randn(2, 4, 8, 8, 8)
    m = torch.ones(2, 1, 8, 8, 8)
    mse = LatentMetrics.mse(z, z, m)
    assert torch.allclose(mse, torch.zeros(2))


@pytest.mark.unit
def test_latent_l1_zero_on_identical_inputs() -> None:
    z = torch.randn(2, 4, 8, 8, 8)
    m = torch.ones(2, 1, 8, 8, 8)
    l1 = LatentMetrics.l1(z, z, m)
    assert torch.allclose(l1, torch.zeros(2))


@pytest.mark.unit
def test_latent_cosine_one_on_identical_inputs() -> None:
    z = torch.randn(2, 4, 8, 8, 8)
    m = torch.ones(2, 1, 8, 8, 8)
    cos = LatentMetrics.cosine(z, z, m)
    assert torch.allclose(cos, torch.ones(2), atol=1e-5)


@pytest.mark.unit
def test_latent_mse_respects_mask() -> None:
    z_pred = torch.zeros(1, 1, 4, 4, 4)
    z_target = torch.ones(1, 1, 4, 4, 4)
    m = torch.zeros(1, 1, 4, 4, 4)
    m[0, 0, 0, 0, 0] = 1.0   # single voxel inside region
    mse = LatentMetrics.mse(z_pred, z_target, m)
    # only one voxel contributes; (0 - 1)^2 = 1
    assert torch.allclose(mse, torch.ones(1))


@pytest.mark.unit
def test_image_psnr_high_when_inputs_close() -> None:
    pred = torch.zeros(1, 1, 16, 16, 16)
    target = torch.zeros(1, 1, 16, 16, 16)
    mask = torch.ones(1, 1, 16, 16, 16)
    m = ImageMetrics(data_range=1.0)
    psnr = m.psnr(pred, target, mask)
    # Identical → mse→0 → PSNR → +∞ (clamped large in our impl)
    assert torch.isfinite(psnr).all()
    assert psnr.item() > 60.0


@pytest.mark.unit
def test_image_ssim_one_on_identical_inputs() -> None:
    pred = torch.rand(1, 1, 16, 16, 16)
    target = pred.clone()
    mask = torch.ones(1, 1, 16, 16, 16)
    m = ImageMetrics(data_range=1.0)
    ssim = m.ssim(pred, target, mask)
    assert torch.allclose(ssim, torch.ones(1), atol=1e-4)

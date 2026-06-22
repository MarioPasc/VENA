"""Unit tests for S1 v3 per-region image metrics + region derivation.

Pins:

* The new derived regions (netc/ed/et/brain_not_wt) appear in ``RegionMasks``
  when ``m_tumor`` + ``m_brain`` are in the batch.
* ``ImageMetrics.metrics_per_region`` emits PSNR/SSIM/MAE/MSE per region.
* Empty regions ⇒ NaN (per-batch entry).
* MAE² ≤ MSE per voxel; consistency invariant.
* The ``brain`` field is exposed as ``"whole"`` in the per-region dict.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.unit

from vena.model.fm.metrics import ImageMetrics, RegionResolver
from vena.model.fm.metrics.regions import RegionSpec


def _default_specs() -> dict[str, RegionSpec]:
    return {
        "brain": RegionSpec(source="latents_h5", h5_key="m_brain"),
        "wt": RegionSpec(source="derived_from_tumor_latent", threshold=0.5),
        "wt_dilated": RegionSpec(
            source="derived_via_scipy_binary_dilation", structure="ones_3x3x3"
        ),
        "bg": RegionSpec(source="derived"),
        "vessel": RegionSpec(source="skipped"),
    }


def _toy_batch() -> dict[str, torch.Tensor]:
    """Build a batch with brain + per-class tumour masks."""
    B, H, W, D = 1, 4, 4, 4
    m_brain = torch.zeros(B, 1, H, W, D)
    m_brain[..., 1:3, 1:3, 1:3] = 1.0  # 2x2x2 brain block
    # Per-class soft masks, threshold = 0.5
    m_tumor = torch.zeros(B, 3, H, W, D)
    m_tumor[:, 0, 1, 1, 1] = 1.0  # NETC at (1,1,1)
    m_tumor[:, 1, 2, 2, 2] = 1.0  # ED   at (2,2,2)
    m_tumor[:, 2, 1, 2, 1] = 1.0  # ET   at (1,2,1)
    # wt = soft sum >= 0.5 — matches the dataset
    soft_union = torch.clamp(m_tumor.sum(dim=1, keepdim=True), 0.0, 1.0)
    m_wt = (soft_union >= 0.5).float()
    return {
        "m_brain": m_brain,
        "m_tumor": m_tumor,
        "m_wt": m_wt,
    }


def test_region_resolver_populates_v3_derived_regions() -> None:
    """netc/ed/et/brain_not_wt are filled from m_tumor + m_brain."""
    rr = RegionResolver(_default_specs())
    masks = rr.resolve(_toy_batch())
    for name in ("netc", "ed", "et", "brain_not_wt"):
        assert masks.get(name) is not None, f"{name} should be populated"
    # netc has exactly one voxel set, masked by brain (which covers it).
    assert masks.netc.sum().item() == 1
    assert masks.ed.sum().item() == 1
    assert masks.et.sum().item() == 1


def test_region_resolver_v3_regions_disjoint() -> None:
    """The per-sub-region masks must not overlap inside the brain."""
    rr = RegionResolver(_default_specs())
    masks = rr.resolve(_toy_batch())
    overlap_netc_ed = (masks.netc & masks.ed).sum().item()
    overlap_netc_et = (masks.netc & masks.et).sum().item()
    overlap_ed_et = (masks.ed & masks.et).sum().item()
    assert overlap_netc_ed == 0
    assert overlap_netc_et == 0
    assert overlap_ed_et == 0


def test_region_resolver_brain_not_wt_excludes_tumor_voxels() -> None:
    """brain_not_wt = brain & ~(any tumour channel ≥ τ)."""
    rr = RegionResolver(_default_specs())
    batch = _toy_batch()
    masks = rr.resolve(batch)
    brain = masks.brain
    # brain has 8 voxels (2x2x2); 3 are tumour ⇒ 5 are brain_not_wt
    assert brain.sum().item() == 8
    assert masks.brain_not_wt.sum().item() == 8 - 3


def test_region_resolver_v3_derived_absent_when_m_tumor_missing() -> None:
    """Without m_tumor, the derived regions stay None (no crash)."""
    rr = RegionResolver(_default_specs())
    # batch without m_tumor (legacy contract); still has m_wt + m_brain.
    batch = _toy_batch()
    del batch["m_tumor"]
    masks = rr.resolve(batch)
    for name in ("netc", "ed", "et", "brain_not_wt"):
        assert masks.get(name) is None


def test_metrics_per_region_returns_all_four_metrics_per_region() -> None:
    """psnr_db / ssim / mae / mse per region in one call."""
    rr = RegionResolver(_default_specs())
    masks = rr.resolve(_toy_batch())
    pred = torch.zeros(1, 1, 4, 4, 4)
    target = torch.zeros(1, 1, 4, 4, 4)
    target[..., 1, 1, 1] = 0.5  # mismatch only at NETC voxel
    im = ImageMetrics(data_range=1.0, ssim_window_size=3)
    per_region = im.metrics_per_region(pred, target, masks)
    # The "brain" region is renamed to "whole".
    assert "whole" in per_region
    assert "wt" in per_region
    assert "netc" in per_region
    for region_dict in per_region.values():
        for k in ("psnr_db", "ssim", "mae", "mse"):
            assert k in region_dict, f"missing metric {k}"


def test_metrics_per_region_empty_region_returns_nan() -> None:
    """An empty region yields NaN (per-batch entry)."""
    rr = RegionResolver(_default_specs())
    batch = _toy_batch()
    # Zero out the tumour mask: NETC/ED/ET become empty.
    batch["m_tumor"] = torch.zeros_like(batch["m_tumor"])
    batch["m_wt"] = torch.zeros_like(batch["m_wt"])
    masks = rr.resolve(batch)
    pred = torch.zeros(1, 1, 4, 4, 4)
    target = torch.zeros(1, 1, 4, 4, 4)
    im = ImageMetrics(data_range=1.0, ssim_window_size=3)
    per_region = im.metrics_per_region(pred, target, masks)
    for region_name in ("netc", "ed", "et"):
        assert math.isnan(per_region[region_name]["psnr_db"].item())
        assert math.isnan(per_region[region_name]["mae"].item())
        assert math.isnan(per_region[region_name]["mse"].item())


def test_mae_squared_le_mse() -> None:
    """For non-degenerate inputs, MAE² ≤ MSE (Cauchy-Schwarz / Jensen)."""
    rr = RegionResolver(_default_specs())
    masks = rr.resolve(_toy_batch())
    pred = torch.zeros(1, 1, 4, 4, 4)
    target = torch.zeros(1, 1, 4, 4, 4)
    target[..., 1, 1, 1] = 0.3
    target[..., 2, 2, 2] = -0.4
    target[..., 1, 2, 1] = 0.1
    im = ImageMetrics(data_range=1.0, ssim_window_size=3)
    per_region = im.metrics_per_region(pred, target, masks)
    for region_name, mvals in per_region.items():
        mae = mvals["mae"].item()
        mse = mvals["mse"].item()
        if math.isnan(mae) or math.isnan(mse):
            continue
        assert mae * mae <= mse + 1e-6, f"{region_name}: MAE²={mae**2} > MSE={mse}"


def test_metrics_per_region_psnr_matches_manual_computation() -> None:
    """Single-voxel error ⇒ analytical PSNR matches."""
    rr = RegionResolver(_default_specs())
    masks = rr.resolve(_toy_batch())
    pred = torch.zeros(1, 1, 4, 4, 4)
    target = torch.zeros(1, 1, 4, 4, 4)
    target[..., 1, 1, 1] = 0.5  # error of 0.5 at NETC voxel; netc is 1 voxel
    im = ImageMetrics(data_range=1.0, ssim_window_size=3)
    per_region = im.metrics_per_region(pred, target, masks)
    # Manual PSNR for the NETC region: 1 voxel, error 0.5, data_range 1.0.
    # MSE = 0.25, PSNR = 10*log10(1.0/0.25) = 10*log10(4) ≈ 6.020599913
    expected_netc_psnr = 10.0 * math.log10(1.0 / 0.25)
    actual_netc_psnr = per_region["netc"]["psnr_db"].item()
    assert abs(actual_netc_psnr - expected_netc_psnr) < 1e-4, (
        f"NETC PSNR manual={expected_netc_psnr} actual={actual_netc_psnr}"
    )


def test_metrics_per_region_uses_brain_alias_whole() -> None:
    """The full brain mask appears under the ``whole`` key, not ``brain``."""
    rr = RegionResolver(_default_specs())
    masks = rr.resolve(_toy_batch())
    pred = torch.zeros(1, 1, 4, 4, 4)
    target = torch.zeros(1, 1, 4, 4, 4)
    im = ImageMetrics(data_range=1.0, ssim_window_size=3)
    per_region = im.metrics_per_region(pred, target, masks)
    assert "whole" in per_region
    assert "brain" not in per_region  # renamed

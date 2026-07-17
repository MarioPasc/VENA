"""Unit tests for vena.validation.metrics_paired.

Tests cover:
- Individual metric functions with synthetic volumes
- ScanMetrics.to_flat_dict() completeness
- compute_paired_metrics() end-to-end via ScanSample
- ZGD formula correctness
- SSIM-map approach (principled region averaging)
- MS-SSIM NaN guard (min_dim < 90)
- C0-Identity canary: zero MAE when pred == real
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from vena.validation.io import ScanSample
from vena.validation.metrics_paired import (
    MetricConfig,
    _masked_mae,
    _masked_psnr,
    _masked_rmse,
    compute_paired_metrics,
    ms_ssim_brain,
    ms_ssim_wt_bbox,
    ssim_in_region,
    ssim_map_3d,
    zgd,
)

pytestmark = pytest.mark.validation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_H, _W, _D = 32, 32, 32  # big enough for SSIM 7-window with trim


def _ones_brain(shape: tuple[int, int, int] = (_H, _W, _D)) -> np.ndarray:
    """Full-volume brain mask (all True)."""
    return np.ones(shape, dtype=bool)


def _small_wt(shape: tuple[int, int, int] = (_H, _W, _D)) -> np.ndarray:
    """Small WT mask occupying the central 4×4×4 cube."""
    wt = np.zeros(shape, dtype=bool)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    wt[cx - 2 : cx + 2, cy - 2 : cy + 2, cz - 2 : cz + 2] = True
    return wt


def _make_sample(
    pred: np.ndarray,
    real: np.ndarray,
    *,
    brain: np.ndarray | None = None,
    wt: np.ndarray | None = None,
    scan_id: str = "s0",
) -> ScanSample:
    shape = pred.shape
    pred_f32 = pred.astype(np.float32)
    return ScanSample(
        scan_id=scan_id,
        patient_id="p0",
        cohort="TestCohort",
        ring="A",
        method="TEST",
        nfe=1,
        pred=pred_f32,
        pred_raw=pred_f32,  # audit-only; irrelevant for metric tests
        pred_harmonised=pred_f32,  # audit-only; irrelevant for metric tests
        pred_mode="raw",
        raw_p995=float(np.percentile(pred_f32, 99.5)),
        real=real.astype(np.float32),
        brain=brain if brain is not None else _ones_brain(shape),
        wt=wt if wt is not None else _small_wt(shape),
        inference_seconds=1.0,
        peak_vram_mb=100.0,
    )


# ---------------------------------------------------------------------------
# _masked_mae
# ---------------------------------------------------------------------------


def test_masked_mae_identity() -> None:
    """MAE is 0 when pred == real."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    mask = _ones_brain()
    assert _masked_mae(vol, vol, mask) == pytest.approx(0.0, abs=1e-6)


def test_masked_mae_constant_offset() -> None:
    """MAE equals the constant offset when pred = real + c."""
    real = np.zeros((_H, _W, _D), dtype=np.float32)
    pred = np.full((_H, _W, _D), 0.25, dtype=np.float32)
    mask = _ones_brain()
    assert _masked_mae(pred, real, mask) == pytest.approx(0.25, rel=1e-5)


def test_masked_mae_empty_mask_nan() -> None:
    """Empty mask yields NaN (no voxels to average over)."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    mask = np.zeros((_H, _W, _D), dtype=bool)
    result = _masked_mae(vol, vol, mask)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# _masked_rmse
# ---------------------------------------------------------------------------


def test_masked_rmse_identity() -> None:
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    mask = _ones_brain()
    assert _masked_rmse(vol, vol, mask) == pytest.approx(0.0, abs=1e-6)


def test_masked_rmse_constant_offset() -> None:
    """RMSE of a constant offset c equals c exactly."""
    real = np.zeros((_H, _W, _D), dtype=np.float32)
    pred = np.full((_H, _W, _D), 0.5, dtype=np.float32)
    mask = _ones_brain()
    assert _masked_rmse(pred, real, mask) == pytest.approx(0.5, rel=1e-5)


# ---------------------------------------------------------------------------
# _masked_psnr
# ---------------------------------------------------------------------------


def test_masked_psnr_identity_is_inf() -> None:
    """PSNR of identical volumes is +inf (zero MSE)."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    mask = _ones_brain()
    result = _masked_psnr(vol, vol, mask)
    assert math.isinf(result) and result > 0


def test_masked_psnr_positive_for_noisy() -> None:
    """PSNR of noisy volume is finite and positive."""
    real = _RNG.random((_H, _W, _D)).astype(np.float32)
    pred = (real + 0.05 * _RNG.standard_normal((_H, _W, _D))).clip(0, 1).astype(np.float32)
    mask = _ones_brain()
    result = _masked_psnr(pred, real, mask)
    assert math.isfinite(result)
    assert result > 0


# ---------------------------------------------------------------------------
# ssim_map_3d and ssim_in_region
# ---------------------------------------------------------------------------


def test_ssim_map_identity() -> None:
    """SSIM map for identical volumes is all 1 inside the valid region."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    s_map = ssim_map_3d(vol, vol)
    # All values should be very close to 1
    np.testing.assert_allclose(s_map, np.ones_like(s_map), atol=1e-4)


def test_ssim_map_shape_is_trimmed() -> None:
    """SSIM map spatial dims are H - (k-1), i.e. trimmed by k//2 per side."""
    k = 7
    s_map = ssim_map_3d(
        _RNG.random((_H, _W, _D)).astype(np.float32),
        _RNG.random((_H, _W, _D)).astype(np.float32),
        window_size=k,
    )
    trim = k // 2
    expected_shape = (_H - 2 * trim, _W - 2 * trim, _D - 2 * trim)
    assert s_map.shape == expected_shape, f"Expected {expected_shape}, got {s_map.shape}"


def test_ssim_in_region_identity_is_one() -> None:
    """ssim_in_region on identical volumes with full mask returns ~1."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    s_map = ssim_map_3d(vol, vol, window_size=7)
    brain = _ones_brain((_H, _W, _D))
    result = ssim_in_region(s_map, brain, window_size=7)
    assert result == pytest.approx(1.0, abs=1e-4)


def test_ssim_in_region_range() -> None:
    """ssim_in_region on random volumes is in (-1, 1]."""
    pred = _RNG.random((_H, _W, _D)).astype(np.float32)
    real = _RNG.random((_H, _W, _D)).astype(np.float32)
    s_map = ssim_map_3d(pred, real, window_size=7)
    brain = _ones_brain((_H, _W, _D))
    result = ssim_in_region(s_map, brain, window_size=7)
    assert -1.0 <= result <= 1.0


def test_ssim_in_region_empty_mask_nan() -> None:
    """Empty trimmed mask yields NaN."""
    # Use a wt mask that is entirely within the trim band: it won't survive
    # the center-crop and the trimmed mask will be all-False.
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    s_map = ssim_map_3d(vol, vol, window_size=7)
    # Mask that is only True at the very edge (will be cropped out)
    mask = np.zeros((_H, _W, _D), dtype=bool)
    mask[0, 0, 0] = True  # edge voxel — falls in the trim band for k=7 (trim=3)
    result = ssim_in_region(s_map, mask, window_size=7)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# ZGD
# ---------------------------------------------------------------------------


def test_zgd_identity_is_one() -> None:
    """ZGD of identical volumes is 1 (ratio of equal gradients)."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    brain = _ones_brain()
    result = zgd(vol, vol, brain)
    assert result == pytest.approx(1.0, abs=1e-4)


def test_zgd_constant_volume() -> None:
    """Constant volume has zero z-gradient; ZGD is NaN (0/0 protected)."""
    vol = np.full((_H, _W, _D), 0.5, dtype=np.float32)
    brain = _ones_brain()
    result = zgd(vol, vol, brain)
    # Both numerator and denominator are 0; result should be NaN or 1
    assert math.isnan(result) or result == pytest.approx(1.0, abs=1e-4)


def test_zgd_smoother_pred_below_one() -> None:
    """A smoother (blurred) prediction should have ZGD < 1 vs the sharp real."""
    rng = np.random.default_rng(42)
    real = rng.random((_H, _W, _D)).astype(np.float32)
    # Smooth prediction by averaging along z
    pred = real.copy()
    pred[..., 1:-1] = (real[..., :-2] + real[..., 1:-1] + real[..., 2:]) / 3.0
    brain = _ones_brain()
    result = zgd(pred, real, brain)
    assert math.isfinite(result)
    # The blurred version should have smaller z-gradients → ZGD < 1
    assert result < 1.0


# ---------------------------------------------------------------------------
# MS-SSIM guards
# ---------------------------------------------------------------------------


def test_ms_ssim_brain_returns_float_no_raise() -> None:
    """ms_ssim_brain always returns a float (finite or NaN) — never raises.

    Whether MONAI succeeds depends on the version-specific minimum volume size,
    so we only assert no exception and a float return type.  The real evaluation
    volumes (≥240 voxels per dim) will always be finite.
    """
    big = np.random.default_rng(7).random((96, 96, 96)).astype(np.float32)
    brain = np.ones((96, 96, 96), dtype=bool)
    weights = (0.0448, 0.2856, 0.3001, 0.3633)
    result = ms_ssim_brain(big, big, brain, weights=weights)
    # Must return float (finite or NaN), never raise
    assert isinstance(result, float), f"Expected float, got {type(result)}"


def test_ms_ssim_brain_tiny_volume_returns_nan() -> None:
    """ms_ssim_brain on 32^3 volumes returns NaN (below MONAI min_dim=90)."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    brain = _ones_brain()
    weights = (0.0448, 0.2856, 0.3001, 0.3633)
    result = ms_ssim_brain(vol, vol, brain, weights=weights)
    # 32 < 90 — MONAI raises, implementation must catch and return NaN
    assert math.isnan(result), f"Expected NaN for 32^3 volume (below MONAI min_dim), got {result}"


def test_ms_ssim_wt_bbox_small_returns_nan() -> None:
    """ms_ssim_wt_bbox returns NaN when the WT bbox is smaller than min_dim=90."""
    # _H=_W=_D=32 — any crop will be <<90 voxels per dim
    pred = _RNG.random((_H, _W, _D)).astype(np.float32)
    real = _RNG.random((_H, _W, _D)).astype(np.float32)
    wt = _small_wt()
    weights = (0.0448, 0.2856, 0.3001, 0.3633)
    result = ms_ssim_wt_bbox(pred, real, wt, weights=weights, min_dim=90)
    assert math.isnan(result), f"Expected NaN for tiny wt bbox, got {result}"


# ---------------------------------------------------------------------------
# compute_paired_metrics — C0-Identity canary
# ---------------------------------------------------------------------------


def test_compute_paired_metrics_c0_identity() -> None:
    """C0-Identity (pred == real) must have zero MAE across all regions."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    brain = _ones_brain()
    wt = _small_wt()
    scan = _make_sample(vol, vol, brain=brain, wt=wt)
    cfg = MetricConfig()
    m = compute_paired_metrics(scan, cfg)
    assert m.mae_brain == pytest.approx(0.0, abs=1e-6)
    assert m.mae_wt == pytest.approx(0.0, abs=1e-6)
    assert m.mae_bg_undilated == pytest.approx(0.0, abs=1e-6)
    assert m.rmse_brain == pytest.approx(0.0, abs=1e-6)
    assert m.ssim_brain == pytest.approx(1.0, abs=1e-3)
    assert m.zgd == pytest.approx(1.0, abs=1e-3)


def test_compute_paired_metrics_to_flat_dict_completeness() -> None:
    """to_flat_dict() contains all expected keys including identity fields."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    scan = _make_sample(vol, vol)
    cfg = MetricConfig()
    m = compute_paired_metrics(scan, cfg)
    d = m.to_flat_dict()

    required = [
        "scan_id",
        "patient_id",
        "method",
        "cohort",
        "ring",
        "nfe",
        "inference_seconds",
        "peak_vram_mb",
        "mae_brain",
        "mae_wt",
        "mae_bg_undilated",
        "rmse_brain",
        "rmse_wt",
        "rmse_bg_undilated",
        "psnr_brain",
        "psnr_wt",
        "psnr_bg_undilated",
        "ssim_brain",
        "ssim_wt",
        "ssim_bg_undilated",
        "ms_ssim_brain",
        "ms_ssim_wt",
        "ms_ssim_bg_undilated",
        "zgd",
        "n_brain_voxels",
        "n_wt_voxels",
        "n_bg_undilated_voxels",
    ]
    missing = [k for k in required if k not in d]
    assert not missing, f"Missing keys in to_flat_dict(): {missing}"


def test_compute_paired_metrics_voxel_counts() -> None:
    """Voxel count fields are positive integers matching mask sizes."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    brain = _ones_brain()
    wt = _small_wt()
    scan = _make_sample(vol, vol, brain=brain, wt=wt)
    cfg = MetricConfig()
    m = compute_paired_metrics(scan, cfg)

    # brain mask is all-ones
    assert m.n_brain_voxels == _H * _W * _D
    # wt mask is 4×4×4 = 64 voxels
    assert m.n_wt_voxels == 64
    # bg_undilated = brain \ wt_dilated, so smaller
    assert m.n_bg_undilated_voxels < _H * _W * _D
    assert m.n_bg_undilated_voxels > 0


def test_compute_paired_metrics_scan_id_propagated() -> None:
    """scan_id/patient_id/method/cohort/ring/nfe round-trip through ScanMetrics."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    scan = ScanSample(
        scan_id="mysc",
        patient_id="mypt",
        cohort="MyCohort",
        ring="B",
        method="MyMethod",
        nfe=7,
        pred=vol,
        pred_raw=vol,
        pred_harmonised=vol,
        pred_mode="raw",
        raw_p995=float(np.percentile(vol, 99.5)),
        real=vol,
        brain=_ones_brain(),
        wt=_small_wt(),
        inference_seconds=3.14,
        peak_vram_mb=512.0,
    )
    cfg = MetricConfig()
    m = compute_paired_metrics(scan, cfg)
    assert m.scan_id == "mysc"
    assert m.patient_id == "mypt"
    assert m.cohort == "MyCohort"
    assert m.ring == "B"
    assert m.method == "MyMethod"
    assert m.nfe == 7
    assert m.inference_seconds == pytest.approx(3.14)
    assert m.peak_vram_mb == pytest.approx(512.0)
    # §4.1 audit fields must round-trip through ScanMetrics
    assert m.pred_mode == "raw"
    assert math.isfinite(m.raw_p995)


def test_scan_metrics_is_frozen() -> None:
    """ScanMetrics is frozen — mutation raises FrozenInstanceError."""
    vol = _RNG.random((_H, _W, _D)).astype(np.float32)
    scan = _make_sample(vol, vol)
    m = compute_paired_metrics(scan, MetricConfig())
    # Name the type: a blind Exception would also pass if the attribute simply
    # did not exist, so it would not actually prove frozenness.
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.mae_brain = 999.0  # type: ignore[misc]

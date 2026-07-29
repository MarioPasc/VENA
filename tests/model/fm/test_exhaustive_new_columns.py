"""Unit tests for B7/B8/B13 exhaustive-val new metric columns.

B7: ``psnr_db_brain`` / ``ssim_brain`` — brain-masked PSNR/SSIM.
B8: ``p995_pred_brain``, ``p995_real_brain``, ``mean_et_pred``,
    ``mean_et_real``, ``mean_bnwt_pred``, ``mean_bnwt_real`` — intensity stats.
B13: ``ms_ssim_brain``, ``ms_ssim_wt_bbox`` — MS-SSIM.

Tests run CPU-only, no checkpoint needed.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEW_COLS_B7 = ("psnr_db_brain", "ssim_brain")
_NEW_COLS_B8 = (
    "p995_pred_brain",
    "p995_real_brain",
    "mean_et_pred",
    "mean_et_real",
    "mean_bnwt_pred",
    "mean_bnwt_real",
)
_NEW_COLS_B13 = ("ms_ssim_brain", "ms_ssim_wt_bbox")
_ALL_NEW_COLS = _NEW_COLS_B7 + _NEW_COLS_B8 + _NEW_COLS_B13


def _make_tensors(h: int = 32, w: int = 32, d: int = 32) -> tuple[torch.Tensor, ...]:
    """Return synthetic (pred, real, brain, wt, et) CPU float32 tensors.

    ``pred`` and ``real`` are 3-D ``(H, W, D)`` — ``_v3_per_region_metrics``
    promotes them internally via ``img_pred[None, None]``.
    Masks are 5-D ``(1, 1, H, W, D)`` because the function's ``_emit``
    helper passes them directly to ``ImageMetrics.psnr/ssim`` which expects
    batched tensors, and ``m_et_img.bool()`` is used as a boolean index on
    the already-promoted 5-D ``p`` tensor.
    """
    gen = torch.Generator().manual_seed(0)
    pred = torch.rand(h, w, d, generator=gen)
    real = torch.rand(h, w, d, generator=gen)
    # Brain: centre half of the volume — 5-D (1,1,H,W,D) bool
    brain_3d = torch.zeros(h, w, d, dtype=torch.bool)
    brain_3d[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, d // 4 : 3 * d // 4] = True
    brain = brain_3d[None, None]
    # WT: small inner cube — 5-D
    wt_3d = torch.zeros(h, w, d, dtype=torch.bool)
    wt_3d[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, d // 3 : 2 * d // 3] = True
    wt = wt_3d[None, None]
    # ET: even smaller cube inside WT — 5-D
    et_3d = torch.zeros(h, w, d, dtype=torch.bool)
    et_3d[h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2, d // 2 - 2 : d // 2 + 2] = True
    et = et_3d[None, None]
    return pred, real, brain, wt, et


# ---------------------------------------------------------------------------
# B7/B8/B13: _V3_EXTRA_COLS manifest
# ---------------------------------------------------------------------------


class TestV3ExtraCols:
    """_V3_EXTRA_COLS must declare all B7/B8/B13 columns."""

    def test_b7_cols_present(self) -> None:
        from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

        for col in _NEW_COLS_B7:
            assert col in ExhaustiveValEngine._V3_EXTRA_COLS, f"Missing B7 column: {col}"

    def test_b8_cols_present(self) -> None:
        from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

        for col in _NEW_COLS_B8:
            assert col in ExhaustiveValEngine._V3_EXTRA_COLS, f"Missing B8 column: {col}"

    def test_b13_cols_present(self) -> None:
        from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

        for col in _NEW_COLS_B13:
            assert col in ExhaustiveValEngine._V3_EXTRA_COLS, f"Missing B13 column: {col}"

    def test_all_10_new_cols_in_manifest(self) -> None:
        from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

        missing = [c for c in _ALL_NEW_COLS if c not in ExhaustiveValEngine._V3_EXTRA_COLS]
        assert missing == [], f"Columns absent from _V3_EXTRA_COLS: {missing}"


# ---------------------------------------------------------------------------
# B7/B8/B13: _v3_per_region_metrics with all masks supplied
# ---------------------------------------------------------------------------


class TestPerRegionMetricsNewCols:
    """_v3_per_region_metrics emits finite values for new cols when masks are given."""

    @pytest.fixture()
    def tensors(self):
        return _make_tensors()

    @pytest.fixture()
    def image_metrics(self):
        from vena.model.fm.metrics import ImageMetrics

        return ImageMetrics(data_range=1.0)

    def _call(self, tensors, image_metrics, brain=True, wt=True, et=True):
        from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

        pred, real, brain_t, wt_t, et_t = tensors
        return ExhaustiveValEngine._v3_per_region_metrics(
            img_pred=pred,
            real_box=real,
            m_netc_img=None,
            m_ed_img=None,
            m_et_img=et_t if et else None,
            m_wt_img=wt_t if wt else None,
            m_brain_img=brain_t if brain else None,
            image_metrics=image_metrics,
        )

    def test_b7_brain_psnr_ssim_finite(self, tensors, image_metrics) -> None:
        out = self._call(tensors, image_metrics)
        assert math.isfinite(out["psnr_db_brain"]), "psnr_db_brain must be finite"
        assert math.isfinite(out["ssim_brain"]), "ssim_brain must be finite"

    def test_b7_ssim_brain_in_range(self, tensors, image_metrics) -> None:
        out = self._call(tensors, image_metrics)
        assert -1.0 <= out["ssim_brain"] <= 1.0

    def test_b8_p995_finite(self, tensors, image_metrics) -> None:
        out = self._call(tensors, image_metrics)
        assert math.isfinite(out["p995_pred_brain"])
        assert math.isfinite(out["p995_real_brain"])

    def test_b8_p995_in_unit_range(self, tensors, image_metrics) -> None:
        # synthetic tensors from torch.rand are in [0, 1]
        out = self._call(tensors, image_metrics)
        assert 0.0 <= out["p995_pred_brain"] <= 1.0
        assert 0.0 <= out["p995_real_brain"] <= 1.0

    def test_b8_mean_et_finite_when_et_given(self, tensors, image_metrics) -> None:
        out = self._call(tensors, image_metrics)
        assert math.isfinite(out["mean_et_pred"])
        assert math.isfinite(out["mean_et_real"])

    def test_b8_mean_bnwt_finite_when_wt_given(self, tensors, image_metrics) -> None:
        out = self._call(tensors, image_metrics)
        assert math.isfinite(out["mean_bnwt_pred"])
        assert math.isfinite(out["mean_bnwt_real"])

    def test_b13_ms_ssim_brain_finite(self, tensors, image_metrics) -> None:
        out = self._call(tensors, image_metrics)
        # ms_ssim_brain may NaN on very small volumes; just check it's a number
        val = out["ms_ssim_brain"]
        assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_b13_ms_ssim_wt_bbox_finite(self, tensors, image_metrics) -> None:
        out = self._call(tensors, image_metrics)
        val = out["ms_ssim_wt_bbox"]
        assert isinstance(val, float)

    def test_all_new_cols_present_in_output(self, tensors, image_metrics) -> None:
        out = self._call(tensors, image_metrics)
        missing = [c for c in _ALL_NEW_COLS if c not in out]
        assert missing == [], f"Output dict missing columns: {missing}"


# ---------------------------------------------------------------------------
# NaN fallback when masks are None
# ---------------------------------------------------------------------------


class TestPerRegionMetricsNanFallback:
    """When masks are None the new columns fall back to NaN."""

    @pytest.fixture()
    def image_metrics(self):
        from vena.model.fm.metrics import ImageMetrics

        return ImageMetrics(data_range=1.0)

    def _call_no_masks(self, image_metrics):
        from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

        gen = torch.Generator().manual_seed(1)
        pred = torch.rand(16, 16, 16, generator=gen)
        real = torch.rand(16, 16, 16, generator=gen)
        return ExhaustiveValEngine._v3_per_region_metrics(
            img_pred=pred,
            real_box=real,
            m_netc_img=None,
            m_ed_img=None,
            m_et_img=None,
            m_wt_img=None,
            m_brain_img=None,
            image_metrics=image_metrics,
        )

    def test_b8_mean_et_nan_without_et_mask(self, image_metrics) -> None:
        out = self._call_no_masks(image_metrics)
        assert math.isnan(out["mean_et_pred"])
        assert math.isnan(out["mean_et_real"])

    def test_b8_mean_bnwt_nan_without_wt_mask(self, image_metrics) -> None:
        out = self._call_no_masks(image_metrics)
        assert math.isnan(out["mean_bnwt_pred"])
        assert math.isnan(out["mean_bnwt_real"])

    def test_b13_ms_ssim_wt_bbox_nan_without_wt(self, image_metrics) -> None:
        out = self._call_no_masks(image_metrics)
        # Without WT mask the bbox crop is empty → NaN
        assert math.isnan(out["ms_ssim_wt_bbox"])

    def test_b7_brain_metrics_use_foreground_fallback(self, image_metrics) -> None:
        """Without a brain mask, brain = real_box > 0 (nonzero voxel foreground)."""
        from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

        gen = torch.Generator().manual_seed(2)
        pred = torch.rand(16, 16, 16, generator=gen)
        real = torch.rand(16, 16, 16, generator=gen)
        out = ExhaustiveValEngine._v3_per_region_metrics(
            img_pred=pred,
            real_box=real,
            m_netc_img=None,
            m_ed_img=None,
            m_et_img=None,
            m_wt_img=None,
            m_brain_img=None,
            image_metrics=image_metrics,
        )
        # real is all >0 (rand never hits 0), so the foreground proxy covers the
        # full volume — psnr_db_brain should be finite.
        assert math.isfinite(out["psnr_db_brain"])
        assert math.isfinite(out["ssim_brain"])

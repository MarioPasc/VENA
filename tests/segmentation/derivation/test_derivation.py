"""Tests for vena.segmentation.derivation (task 16).

All tests are pure-torch — no checkpoint, no GPU required.
Marker: segmentation (registered in pyproject.toml).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from vena.segmentation.config import DerivationConfig
from vena.segmentation.derivation import (
    ClassTemperatures,
    apply_temperature,
    ensemble_soft,
    fit_temperatures,
    pool_to_latent,
)
from vena.segmentation.exceptions import SegDerivationError

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CROP_H, _CROP_W, _CROP_D = 192, 224, 192  # LATENT_CROP_BOX
_LAT_H, _LAT_W, _LAT_D = 48, 56, 48  # LATENT_SPATIAL


def _default_cfg() -> DerivationConfig:
    return DerivationConfig()


# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------


class TestTemperatureCalibration:
    """fit_temperatures + apply_temperature correctness."""

    def test_fitted_temperature_reduces_nll(self) -> None:
        """Overconfident logits with 50% errors → fitted T > 1 reduces NLL."""
        # logit=5 strongly predicts class 1, but half the volume is class 0.
        logits_wt = torch.full((8, 8, 8), 5.0)
        target_wt = torch.zeros(8, 8, 8)
        target_wt[4:, :, :] = 1.0  # 50% correct, 50% wrong

        # Less extreme overconfidence on channel 1.
        logits_netc = torch.full((8, 8, 8), 2.0)
        target_netc = torch.zeros(8, 8, 8)
        target_netc[4:, :, :] = 1.0

        logits = torch.stack([logits_wt, logits_netc], dim=0)
        target = torch.stack([target_wt, target_netc], dim=0)

        temps = fit_temperatures(logits, target)

        assert temps.t_wt > 0.0
        assert temps.t_netc > 0.0
        # 50% errors with high-confidence logits require T > 1.
        assert temps.t_wt > 1.0, f"Expected T_WT > 1; got {temps.t_wt:.4f}"
        assert temps.t_netc > 1.0, f"Expected T_NETC > 1; got {temps.t_netc:.4f}"

        # NLL must decrease after temperature scaling.
        nll_before = F.binary_cross_entropy_with_logits(logits, target)
        t_tensor = logits.new_tensor([temps.t_wt, temps.t_netc]).view(2, 1, 1, 1)
        nll_after = F.binary_cross_entropy_with_logits(logits / t_tensor, target)
        assert nll_after.item() < nll_before.item(), (
            f"Fitted T must reduce NLL; before={nll_before:.4f}, after={nll_after:.4f}"
        )

    def test_per_class_temperature_differs(self) -> None:
        """Different logit magnitudes with same error pattern → T_WT != T_NETC."""
        n = 8
        # Channel 0 (WT): large logit, ~12.5% wrong → T_WT* = 5/log(7) ≈ 2.57
        logits_wt = torch.full((n, n, n), 5.0)
        target_wt = torch.ones(n, n, n)
        target_wt[0, :, :] = 0.0  # first H-slice wrong

        # Channel 1 (NETC): small logit, same error fraction → T_NETC* = 1/log(7) ≈ 0.51
        logits_netc = torch.full((n, n, n), 1.0)
        target_netc = torch.ones(n, n, n)
        target_netc[0, :, :] = 0.0

        logits = torch.stack([logits_wt, logits_netc], dim=0)
        targets = torch.stack([target_wt, target_netc], dim=0)

        temps = fit_temperatures(logits, targets)

        # Analytical: T_WT*/T_NETC* ≈ 5 → gap >> 0.5
        assert abs(temps.t_wt - temps.t_netc) > 0.5, (
            f"Expected |T_WT - T_NETC| > 0.5; got T_WT={temps.t_wt:.4f}, T_NETC={temps.t_netc:.4f}"
        )

    def test_argmax_preserved(self) -> None:
        """apply_temperature preserves the per-voxel hard decision (threshold at 0.5)."""
        torch.manual_seed(42)
        logits = torch.randn(2, 6, 6, 6) * 2.0
        # Fit on targets matching the raw argmax — T may be small.
        target = (logits > 0).float()
        temps = fit_temperatures(logits, target)

        probs = apply_temperature(logits, temps)

        # Since T > 0, sign(logit) = sign(logit/T) → thresholded decision unchanged.
        assert torch.all((probs > 0.5) == (logits > 0)), (
            "apply_temperature must preserve the thresholded (argmax) decision"
        )
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0

    def test_apply_temperature_output_range(self) -> None:
        """Output of apply_temperature is always in [0, 1]."""
        torch.manual_seed(7)
        logits = torch.randn(2, 4, 4, 4) * 10.0
        temps = ClassTemperatures(t_wt=2.0, t_netc=0.5)
        probs = apply_temperature(logits, temps)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0
        assert probs.shape == logits.shape

    def test_shape_mismatch_raises(self) -> None:
        """Incompatible logit/target shapes → SegDerivationError."""
        logits = torch.randn(2, 4, 4, 4)
        target = torch.zeros(2, 3, 4, 4)  # wrong H
        with pytest.raises(SegDerivationError):
            fit_temperatures(logits, target)

    def test_wrong_channel_count_raises(self) -> None:
        """First dim != 2 → SegDerivationError."""
        logits = torch.randn(3, 4, 4, 4)
        target = torch.zeros(3, 4, 4, 4)
        with pytest.raises(SegDerivationError):
            fit_temperatures(logits, target)

    def test_nonpositive_temperature_raises(self) -> None:
        """Non-positive T in ClassTemperatures → SegDerivationError from apply_temperature."""
        logits = torch.randn(2, 4, 4, 4)
        with pytest.raises(SegDerivationError):
            apply_temperature(logits, ClassTemperatures(t_wt=0.0, t_netc=1.0))
        with pytest.raises(SegDerivationError):
            apply_temperature(logits, ClassTemperatures(t_wt=1.0, t_netc=-0.5))


# ---------------------------------------------------------------------------
# Partial-volume pooling
# ---------------------------------------------------------------------------


class TestPartialVolumePooling:
    """pool_to_latent: partial-volume semantics and output range."""

    def test_half_filled_block(self) -> None:
        """A half-filled 4×4×4 cell → latent value ≈ 0.5."""
        cfg = _default_cfg()
        stride = cfg.avg_pool_stride  # 4

        prob = torch.zeros(2, _CROP_H, _CROP_W, _CROP_D)
        # Latent cell [0, 0, 0] covers image voxels [0:4, 0:4, 0:4].
        # Fill half of H (2 out of 4 voxels) → avg = 2/4 = 0.5.
        prob[0, 0:2, 0:stride, 0:stride] = 1.0

        result = pool_to_latent(prob, cfg)

        val = result[0, 0, 0, 0].item()
        assert abs(val - 0.5) < 1e-5, f"Half-filled cell: expected 0.5, got {val}"

    def test_fully_inside(self) -> None:
        """A fully filled 4×4×4 cell → latent value ≈ 1.0."""
        cfg = _default_cfg()
        stride = cfg.avg_pool_stride

        prob = torch.zeros(2, _CROP_H, _CROP_W, _CROP_D)
        prob[0, stride : 2 * stride, 0:stride, 0:stride] = 1.0  # cell [1, 0, 0]

        result = pool_to_latent(prob, cfg)
        val = result[0, 1, 0, 0].item()
        assert abs(val - 1.0) < 1e-5, f"Fully inside: expected 1.0, got {val}"

    def test_fully_outside(self) -> None:
        """A fully empty volume → all latent values ≈ 0.0."""
        cfg = _default_cfg()
        prob = torch.zeros(2, _CROP_H, _CROP_W, _CROP_D)
        result = pool_to_latent(prob, cfg)
        assert result.abs().max().item() < 1e-6, "Fully empty: expected all zeros"

    def test_values_in_unit_interval(self) -> None:
        """Output values are in [0, 1] when input is in [0, 1]."""
        cfg = _default_cfg()
        torch.manual_seed(7)
        prob = torch.rand(2, _CROP_H, _CROP_W, _CROP_D)
        result = pool_to_latent(prob, cfg)
        assert result.min().item() >= 0.0, "Output must be >= 0"
        assert result.max().item() <= 1.0, "Output must be <= 1"


# ---------------------------------------------------------------------------
# Grid + registration
# ---------------------------------------------------------------------------


class TestGridAndRegistration:
    """pool_to_latent output grid and voxel-level registration."""

    def test_output_shape_exact(self) -> None:
        """Output is exactly (2, 48, 56, 48)."""
        cfg = _default_cfg()
        prob = torch.zeros(2, _CROP_H, _CROP_W, _CROP_D)
        result = pool_to_latent(prob, cfg)
        assert result.shape == (2, _LAT_H, _LAT_W, _LAT_D), (
            f"Expected (2, 48, 56, 48); got {tuple(result.shape)}"
        )

    def test_centroid_registration(self) -> None:
        """A block at image centroid (i*4, j*4, k*4) maps to latent cell (i, j, k)."""
        cfg = _default_cfg()
        stride = cfg.avg_pool_stride

        # Place a known 4×4×4 block well inside the grid.
        i, j, k = 20, 24, 20  # latent indices (arbitrary, divisible range)
        hi, wi, di = i * stride, j * stride, k * stride

        prob = torch.zeros(2, _CROP_H, _CROP_W, _CROP_D)
        prob[0, hi : hi + stride, wi : wi + stride, di : di + stride] = 1.0

        result = pool_to_latent(prob, cfg)

        assert result.shape == (2, _LAT_H, _LAT_W, _LAT_D)

        # The exact cell must be 1.0 (fully covered).
        cell_val = result[0, i, j, k].item()
        assert abs(cell_val - 1.0) < 1e-5, (
            f"Centroid block at latent ({i},{j},{k}): expected 1.0, got {cell_val}"
        )

        # Neighbouring cells must be 0 (no leakage).
        if i + 1 < _LAT_H:
            assert result[0, i + 1, j, k].item() < 1e-5
        if j + 1 < _LAT_W:
            assert result[0, i, j + 1, k].item() < 1e-5
        if k + 1 < _LAT_D:
            assert result[0, i, j, k + 1].item() < 1e-5

    def test_with_crop_spec(self) -> None:
        """crop_spec path: native (240,240,155) → (2,48,56,48) output."""
        from vena.common import CropPadSpec

        cfg = _default_cfg()
        native = (240, 240, 155)
        # Typical brain-centred origin so the crop box fits within native.
        crop_origin = (24, 8, 18)
        crop_spec = CropPadSpec(
            crop_origin=crop_origin,
            native_shape=native,
            target_shape=(_CROP_H, _CROP_W, _CROP_D),
        )
        torch.manual_seed(1)
        prob = torch.rand(2, *native)
        result = pool_to_latent(prob, cfg, crop_spec=crop_spec)
        assert result.shape == (2, _LAT_H, _LAT_W, _LAT_D), (
            f"crop_spec path: expected (2,48,56,48), got {tuple(result.shape)}"
        )

    def test_wrong_input_channels_raises(self) -> None:
        """Input with != 2 channels raises SegDerivationError."""
        cfg = _default_cfg()
        with pytest.raises(SegDerivationError):
            pool_to_latent(torch.zeros(3, _CROP_H, _CROP_W, _CROP_D), cfg)

    def test_wrong_ndim_raises(self) -> None:
        """Input with wrong ndim raises SegDerivationError."""
        cfg = _default_cfg()
        with pytest.raises(SegDerivationError):
            pool_to_latent(torch.zeros(2, _CROP_H, _CROP_W), cfg)


# ---------------------------------------------------------------------------
# Order matters: sigmoid first vs pool-then-sigmoid
# ---------------------------------------------------------------------------


class TestOrderMatters:
    """Sigmoid must precede avg-pooling (not follow it)."""

    def test_sigmoid_first_path_no_negatives(self) -> None:
        """Pooling of sigmoid probabilities yields no negative values."""
        cfg = _default_cfg()
        torch.manual_seed(3)
        logits = torch.randn(2, _CROP_H, _CROP_W, _CROP_D) * 3.0
        probs = torch.sigmoid(logits)  # strictly in [0, 1]

        result = pool_to_latent(probs, cfg)
        assert result.min().item() >= 0.0, "Sigmoid-first path must have no negatives"

    def test_sigmoid_first_differs_from_pool_then_sigmoid(self) -> None:
        """E[sigmoid(X)] != sigmoid(E[X]) by Jensen's inequality."""
        cfg = _default_cfg()
        torch.manual_seed(5)
        logits = torch.randn(2, _CROP_H, _CROP_W, _CROP_D) * 3.0

        # Correct path: sigmoid → pool.
        probs = torch.sigmoid(logits)
        pooled_sigmoid_first = pool_to_latent(probs, cfg)

        # Incorrect path: pool raw logits → sigmoid.
        stride = cfg.avg_pool_stride
        x_pooled = F.avg_pool3d(logits, kernel_size=stride, stride=stride)
        pooled_then_sigmoid = torch.sigmoid(x_pooled)

        assert not torch.allclose(pooled_sigmoid_first, pooled_then_sigmoid, atol=1e-3), (
            "sigmoid-first and pool-then-sigmoid must differ (Jensen's inequality)"
        )


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class TestEnsemble:
    """ensemble_soft: mean correctness, variance channel, error handling."""

    _H, _W, _D = _LAT_H, _LAT_W, _LAT_D

    def _identical_maps(self, k: int) -> list[torch.Tensor]:
        base = torch.rand(2, self._H, self._W, self._D)
        return [base] * k

    def test_identical_mean_equals_input(self) -> None:
        """3 identical maps → mean equals the original map."""
        base = torch.rand(2, self._H, self._W, self._D)
        result = ensemble_soft([base, base, base], emit_variance=False)
        assert result.shape == (2, self._H, self._W, self._D)
        assert torch.allclose(result, base, atol=1e-6)

    def test_identical_std_zero(self) -> None:
        """3 identical maps → k-fold disagreement std channel = 0."""
        base = torch.rand(2, self._H, self._W, self._D)
        result = ensemble_soft([base, base, base], emit_variance=True)
        assert result.shape == (3, self._H, self._W, self._D)
        std_channel = result[2:]
        assert std_channel.abs().max().item() < 1e-5, (
            f"Identical maps: std must be 0, got max={std_channel.abs().max():.2e}"
        )

    def test_distinct_std_positive(self) -> None:
        """Distinct maps → k-fold disagreement std > 0 at disagreement voxels."""
        zero_map = torch.zeros(2, self._H, self._W, self._D)
        one_map = torch.ones(2, self._H, self._W, self._D)
        mid_map = torch.full((2, self._H, self._W, self._D), 0.5)

        result = ensemble_soft([zero_map, one_map, mid_map], emit_variance=True)
        assert result.shape == (3, self._H, self._W, self._D)
        std_channel = result[2:]
        assert std_channel.max().item() > 0.1, (
            "Distinct maps must produce non-zero k-fold disagreement"
        )

    def test_emit_variance_false_gives_two_channels(self) -> None:
        """emit_variance=False → 2-channel output."""
        torch.manual_seed(0)
        maps = [torch.rand(2, self._H, self._W, self._D) for _ in range(3)]
        result = ensemble_soft(maps, emit_variance=False)
        assert result.shape[0] == 2, f"Without variance: expected 2 channels, got {result.shape[0]}"

    def test_emit_variance_true_gives_three_channels(self) -> None:
        """emit_variance=True → 3-channel output (mean_wt, mean_netc, disagreement)."""
        torch.manual_seed(1)
        maps = [torch.rand(2, self._H, self._W, self._D) for _ in range(3)]
        result = ensemble_soft(maps, emit_variance=True)
        assert result.shape[0] == 3, f"With variance: expected 3 channels, got {result.shape[0]}"

    def test_mean_spatial_shape_preserved(self) -> None:
        """Ensemble mean has same spatial shape as each input map."""
        torch.manual_seed(2)
        maps = [torch.rand(2, self._H, self._W, self._D) for _ in range(4)]
        result = ensemble_soft(maps)
        assert result.shape == (2, self._H, self._W, self._D)

    def test_single_map_returns_itself(self) -> None:
        """Single-map ensemble mean equals the map; std = 0 (K=1 case)."""
        base = torch.rand(2, self._H, self._W, self._D)
        result_mean = ensemble_soft([base], emit_variance=False)
        assert torch.allclose(result_mean, base, atol=1e-6)

        result_var = ensemble_soft([base], emit_variance=True)
        assert result_var.shape == (3, self._H, self._W, self._D)
        # K=1: std is undefined → implementation returns zeros.
        assert result_var[2:].abs().max().item() < 1e-6

    def test_empty_sequence_raises(self) -> None:
        """Empty input raises SegDerivationError."""
        with pytest.raises(SegDerivationError):
            ensemble_soft([])

    def test_wrong_channel_count_raises(self) -> None:
        """Map with != 2 channels raises SegDerivationError."""
        bad_map = torch.rand(3, self._H, self._W, self._D)
        with pytest.raises(SegDerivationError):
            ensemble_soft([bad_map])

    def test_inconsistent_shapes_raises(self) -> None:
        """Maps with inconsistent shapes raise SegDerivationError."""
        a = torch.rand(2, self._H, self._W, self._D)
        b = torch.rand(2, self._H, self._W, self._D + 4)  # different D
        with pytest.raises(SegDerivationError):
            ensemble_soft([a, b])

"""Tests for vena.segmentation.metrics (task 15).

All 5 acceptance criteria are numeric and self-contained (pure torch/numpy,
no GPU, no checkpoints).
"""

from __future__ import annotations

import math

import pytest
import torch

from vena.segmentation.config import MetricsConfig
from vena.segmentation.metrics import (
    ModelScore,
    average_hausdorff,
    brier,
    check_gseg,
    classwise_ece,
    dice,
    et_diagnostic,
    expected_calibration_error,
    select_ensemble,
)

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ones(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.ones(*shape)


def _zeros(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(*shape)


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — overlap: dice + AHD
# ---------------------------------------------------------------------------


class TestDice:
    def test_identical_returns_one(self) -> None:
        pred = torch.ones(4, 4, 4)
        tgt = torch.ones(4, 4, 4)
        assert dice(pred, tgt) == pytest.approx(1.0)

    def test_disjoint_returns_zero(self) -> None:
        # pred = left half, target = right half, no overlap
        pred = torch.zeros(4, 4, 4)
        pred[:2, :, :] = 1.0
        tgt = torch.zeros(4, 4, 4)
        tgt[2:, :, :] = 1.0
        assert dice(pred, tgt) == pytest.approx(0.0)

    def test_fifty_percent_overlap_hand_value(self) -> None:
        # pred: all 8 voxels = 1  (2x2x2 block)
        # tgt:  first 4 voxels = 1 (one face of the block)
        # |P∩T| = 4, |P| = 8, |T| = 4  → Dice = 2*4/(8+4) = 8/12 = 2/3
        pred = _ones((2, 2, 2))
        tgt = _zeros((2, 2, 2))
        tgt[0, :, :] = 1.0  # 4 voxels
        expected = 2 * 4 / (8 + 4)  # 2/3
        assert dice(pred, tgt) == pytest.approx(expected, abs=1e-6)

    def test_empty_both_returns_one(self) -> None:
        # Both empty → perfect agreement on background
        pred = _zeros((4, 4, 4))
        tgt = _zeros((4, 4, 4))
        assert dice(pred, tgt) == pytest.approx(1.0)

    def test_threshold_applied(self) -> None:
        # Soft pred below threshold → treated as 0
        pred = torch.full((4, 4, 4), 0.4)
        tgt = _ones((4, 4, 4))
        assert dice(pred, tgt, threshold=0.5) == pytest.approx(0.0)


class TestAverageHausdorff:
    def test_identical_returns_zero(self) -> None:
        pred = _ones((1, 1, 4, 4, 4))
        tgt = _ones((1, 1, 4, 4, 4))
        assert average_hausdorff(pred, tgt) == pytest.approx(0.0)

    def test_empty_pred_returns_nan(self) -> None:
        pred = _zeros((4, 4, 4))
        tgt = _ones((4, 4, 4))
        assert math.isnan(average_hausdorff(pred, tgt))

    def test_empty_target_returns_nan(self) -> None:
        pred = _ones((4, 4, 4))
        tgt = _zeros((4, 4, 4))
        assert math.isnan(average_hausdorff(pred, tgt))

    def test_both_empty_returns_nan(self) -> None:
        pred = _zeros((4, 4, 4))
        tgt = _zeros((4, 4, 4))
        assert math.isnan(average_hausdorff(pred, tgt))


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — calibration: ECE + Brier
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_is_zero(self) -> None:
        # All predictions = 0.5, exactly half the targets are 1.
        # In whatever bin 0.5 falls: conf = 0.5, acc = 0.5 → ECE = 0.
        n = 100
        probs = torch.full((n,), 0.5)
        tgt = torch.zeros(n)
        tgt[: n // 2] = 1.0  # 50 positives
        ece = expected_calibration_error(probs, tgt)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_overconfident_is_positive(self) -> None:
        # All predictions = 0.9, all targets = 0 → ECE ≈ 0.9
        probs = torch.full((50,), 0.9)
        tgt = torch.zeros(50)
        ece = expected_calibration_error(probs, tgt)
        assert ece > 0.0

    def test_calibrated_strictly_less_than_overconfident(self) -> None:
        n = 100
        probs_cal = torch.full((n,), 0.5)
        tgt_cal = torch.zeros(n)
        tgt_cal[: n // 2] = 1.0

        probs_over = torch.full((n,), 0.9)
        tgt_over = torch.zeros(n)

        assert expected_calibration_error(probs_cal, tgt_cal) < expected_calibration_error(
            probs_over, tgt_over
        )

    def test_operates_on_raw_probs_not_thresholded(self) -> None:
        # ECE computed at prob=0.3 (below 0.5 threshold) should differ from
        # ECE at prob=0.7; if thresholding were applied both would collapse
        # to 0 or 1 and give the same result.
        probs_low = torch.full((20,), 0.3)
        probs_high = torch.full((20,), 0.7)
        tgt = torch.zeros(20)
        assert expected_calibration_error(probs_low, tgt) != pytest.approx(
            expected_calibration_error(probs_high, tgt)
        )


class TestBrier:
    def test_closed_form_three_values(self) -> None:
        # TC channel: probs=[0.9, 0.3, 0.6], targets=[1, 0, 1]
        #   BS_tc = ((0.9-1)^2 + (0.3-0)^2 + (0.6-1)^2) / 3
        #         = (0.01 + 0.09 + 0.16) / 3 = 0.26/3
        # NETC channel: probs=[0.2, 0.8, 0.5], targets=[0, 1, 0]
        #   BS_netc = ((0.2-0)^2 + (0.8-1)^2 + (0.5-0)^2) / 3
        #           = (0.04 + 0.04 + 0.25) / 3 = 0.33/3
        probs = torch.tensor([[0.9, 0.3, 0.6], [0.2, 0.8, 0.5]])
        tgt = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        result = brier(probs, tgt)

        assert result["tc"] == pytest.approx(0.26 / 3, abs=1e-6)
        assert result["netc"] == pytest.approx(0.33 / 3, abs=1e-6)

    def test_perfect_calibration_zero_brier(self) -> None:
        probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        tgt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = brier(probs, tgt)
        assert result["tc"] == pytest.approx(0.0)
        assert result["netc"] == pytest.approx(0.0)

    def test_operates_on_raw_probs(self) -> None:
        # Brier with prob=0.9 differs from prob=0.1 → not thresholded.
        probs_high = torch.tensor([[0.9], [0.9]])
        probs_low = torch.tensor([[0.1], [0.1]])
        tgt = torch.tensor([[0.0], [0.0]])
        assert brier(probs_high, tgt)["tc"] != pytest.approx(brier(probs_low, tgt)["tc"])


class TestClasswiseEce:
    def test_returns_tc_and_netc_keys(self) -> None:
        probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        tgt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = classwise_ece(probs, tgt)
        assert "tc" in result
        assert "netc" in result

    def test_values_are_floats(self) -> None:
        probs = torch.rand(2, 4, 4, 4)
        tgt = (torch.rand(2, 4, 4, 4) > 0.5).float()
        result = classwise_ece(probs, tgt)
        assert isinstance(result["tc"], float)
        assert isinstance(result["netc"], float)


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — G-SEG gate: check_gseg
# ---------------------------------------------------------------------------


class TestCheckGseg:
    def _cfg(self) -> MetricsConfig:
        return MetricsConfig(gseg_tc_dice=0.75, gseg_netc_dice=0.50)

    def test_exact_thresholds_passes(self) -> None:
        result = check_gseg({"BraTS-GLI": {"tc": 0.75, "netc": 0.50}}, self._cfg())
        assert result.passed is True
        assert result.failures == []

    def test_above_thresholds_passes(self) -> None:
        result = check_gseg({"UCSF": {"tc": 0.85, "netc": 0.65}}, self._cfg())
        assert result.passed is True

    def test_netc_below_threshold_fails(self) -> None:
        result = check_gseg({"ring_b": {"tc": 0.80, "netc": 0.49}}, self._cfg())
        assert result.passed is False
        assert ("ring_b", "netc", 0.49) in result.failures

    def test_tc_below_threshold_fails(self) -> None:
        result = check_gseg({"LUMIERE": {"tc": 0.74, "netc": 0.60}}, self._cfg())
        assert result.passed is False
        assert ("LUMIERE", "tc", 0.74) in result.failures

    def test_both_below_threshold_two_failures(self) -> None:
        result = check_gseg({"bad": {"tc": 0.60, "netc": 0.40}}, self._cfg())
        assert result.passed is False
        assert len(result.failures) == 2

    def test_multi_cohort_one_failing_fails_all(self) -> None:
        cohorts = {
            "A": {"tc": 0.80, "netc": 0.55},
            "B": {"tc": 0.70, "netc": 0.51},  # TC below 0.75
        }
        result = check_gseg(cohorts, self._cfg())
        assert result.passed is False

    def test_healthy_control_all_zero_pred_passes(self) -> None:
        # All-zero prediction → tc_volume = 0.0 → passes the volume check.
        # This verifies the gate uses TC-VOLUME (not Dice) for healthy controls.
        result = check_gseg({"healthy": {"tc_volume": 0.0}}, self._cfg())
        assert result.passed is True
        assert result.failures == []

    def test_healthy_control_large_volume_fails(self) -> None:
        # Model predicts tumour on healthy patient → tc_volume large → fails.
        result = check_gseg({"healthy": {"tc_volume": 0.05}}, self._cfg())
        assert result.passed is False
        assert any(f[1] == "tc_volume" for f in result.failures)

    def test_per_cohort_populated(self) -> None:
        metrics = {"UCSF": {"tc": 0.80, "netc": 0.60}}
        result = check_gseg(metrics, self._cfg())
        assert "UCSF" in result.per_cohort
        assert result.per_cohort["UCSF"]["tc"] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — select_ensemble: dual DSC+Brier selection
# ---------------------------------------------------------------------------


class TestSelectEnsemble:
    def _a(self) -> ModelScore:
        return ModelScore(name="A", dsc=0.85, brier=0.10)

    def _b(self) -> ModelScore:
        return ModelScore(name="B", dsc=0.845, brier=0.06)

    def _b_far(self) -> ModelScore:
        # DSC gap vs A = 0.03 > 0.01 → outside the 1% tolerance
        return ModelScore(name="B_far", dsc=0.82, brier=0.06)

    def test_dual_within_one_percent_picks_better_brier(self) -> None:
        # |DSC_A − DSC_B| = 0.005 < 0.01 → prefer B (lower Brier 0.06 < 0.10)
        winner = select_ensemble([self._a(), self._b()], mode="dual")
        assert winner == "B"

    def test_dual_outside_one_percent_picks_better_dsc(self) -> None:
        # |DSC_A − DSC_B_far| = 0.03 ≥ 0.01 → prefer A (higher DSC 0.85)
        winner = select_ensemble([self._a(), self._b_far()], mode="dual")
        assert winner == "A"

    def test_dice_mode_picks_highest_dsc(self) -> None:
        winner = select_ensemble([self._a(), self._b()], mode="dice")
        assert winner == "A"

    def test_brier_mode_picks_lowest_brier(self) -> None:
        winner = select_ensemble([self._a(), self._b()], mode="brier")
        assert winner == "B"

    def test_default_mode_is_dual(self) -> None:
        # Default mode should behave like "dual"
        winner = select_ensemble([self._a(), self._b()])
        assert winner == "B"

    def test_single_model_returns_it(self) -> None:
        only = ModelScore(name="only", dsc=0.80, brier=0.15)
        assert select_ensemble([only]) == "only"


# ---------------------------------------------------------------------------
# Acceptance criterion 5 — et_diagnostic
# ---------------------------------------------------------------------------


class TestEtDiagnostic:
    def test_returns_required_keys(self) -> None:
        pred = torch.tensor([[0.9, 0.9, 0.1, 0.1], [0.2, 0.2, 0.05, 0.05]])
        tgt = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        result = et_diagnostic(pred, tgt)
        assert "et_dice" in result
        assert "mean_et_soft" in result

    def test_et_dice_perfect_match(self) -> None:
        # ET_pred = clip(TC - NETC, 0, 1) = clip([0.7, 0.7, 0.05, 0.05], 0, 1)
        # thresholded at 0.5 → [1, 1, 0, 0]
        # ET_target = clip([1-0, 1-0, 0-0, 0-0], 0, 1) = [1, 1, 0, 0]
        # → perfect overlap → Dice = 1.0
        pred = torch.tensor([[0.9, 0.9, 0.1, 0.1], [0.2, 0.2, 0.05, 0.05]])
        tgt = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        result = et_diagnostic(pred, tgt)
        assert result["et_dice"] == pytest.approx(1.0)

    def test_mean_et_soft_in_target_region(self) -> None:
        # ET voxels are voxels 0 and 1 (target_et = 1)
        # pred_et = [0.7, 0.7, 0.05, 0.05]
        # mean in target-ET region = (0.7 + 0.7) / 2 = 0.7
        pred = torch.tensor([[0.9, 0.9, 0.1, 0.1], [0.2, 0.2, 0.05, 0.05]])
        tgt = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        result = et_diagnostic(pred, tgt)
        assert result["mean_et_soft"] == pytest.approx(0.7, abs=1e-5)

    def test_empty_target_et_mean_is_nan(self) -> None:
        # NETC == TC everywhere → ET = 0 everywhere
        pred = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        tgt = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # TC == NETC → ET_target = 0
        result = et_diagnostic(pred, tgt)
        assert math.isnan(result["mean_et_soft"])

    def test_et_is_reported_not_gated(self) -> None:
        # et_diagnostic has no interaction with check_gseg — it is pure
        # diagnostic output, verified by the function returning a dict (not
        # raising) regardless of the ET-Dice value.
        pred = torch.zeros(2, 4)  # no predictions
        tgt = torch.ones(2, 4)
        tgt[1, :] = 0.0  # TC=1, NETC=0 everywhere → ET_target = 1
        result = et_diagnostic(pred, tgt)
        # et_dice = 0.0 (no predicted ET), mean_et_soft = 0.0 (pred_et=0 in ET region)
        assert isinstance(result["et_dice"], float)
        assert isinstance(result["mean_et_soft"], float)

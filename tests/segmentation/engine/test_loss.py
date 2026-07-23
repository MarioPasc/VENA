"""Tests for vena.segmentation.engine.loss.

All tests run on CPU with synthetic tensors — no checkpoints, no GPU required.

Acceptance criteria (all 5 numeric):
1. DML on hard labels == 1 - soft_dice(hard labels) to rtol=1e-5.
2. DML properness on soft labels: finite, non-negative, minimal at probs==target,
   strictly increasing under perturbation (Wang-2023 symmetry: DML(p,t)=DML(t,p)).
3. SegmentationLoss.forward → scalar with finite grad; grad ≈ 0 at perfect pred.
4. Deep supervision: 2-head stub weighted sum matches hand computation exactly.
5. Tversky alpha<beta: FN-heavy pred loss > equal-count FP-heavy pred loss.
Plus real-mask sanity via make_soft_targets.
Plus MONAI DiceLoss improperness demonstration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from vena.segmentation.config import LossConfig, TargetConfig
from vena.segmentation.engine.loss import (
    SegmentationLoss,
    _compute_single_loss,
    ce_term,
    dice_semimetric_loss,
    tversky_term,
)
from vena.segmentation.exceptions import SegLossError
from vena.segmentation.targets import make_soft_targets

pytestmark = pytest.mark.segmentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _soft_dice_reference(
    probs: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Inline independent soft-Dice reference for acceptance criterion 1.

    Uses the standard L1 denominator (sum(p) + sum(t)), identical to MONAI
    DiceLoss's default formula.  Matches DML only when both p and t are hard
    (∈ {0,1}).
    """
    b, c = probs.shape[0], probs.shape[1]
    p = probs.reshape(b, c, -1)
    t = target.reshape(b, c, -1)
    inter = (p * t).sum(-1)
    denom = p.sum(-1) + t.sum(-1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


# ---------------------------------------------------------------------------
# Acceptance 1: DML == 1 - soft_dice on HARD labels
# ---------------------------------------------------------------------------


class TestDMLEqualsSOFTDiceOnHardLabels:
    """DML reduces to standard soft-Dice when both probs and target are hard."""

    def test_equality_to_rtol_1e5(self) -> None:
        torch.manual_seed(0)
        shape = (2, 2, 8, 8, 8)
        # Hard probs and hard targets (∈ {0, 1})
        probs_hard = torch.randint(0, 2, shape).float()
        target_hard = torch.randint(0, 2, shape).float()

        dml = dice_semimetric_loss(probs_hard, target_hard, reduction="mean")
        ref = _soft_dice_reference(probs_hard, target_hard)

        max_abs_diff = (dml - ref).abs().item()
        # Report: max_abs_diff should be < rtol * ref ≈ 1e-5 * ref
        assert torch.isclose(dml, ref, rtol=1e-5, atol=1e-7), (
            f"DML={dml.item():.8f} vs soft-Dice={ref.item():.8f}; max_abs_diff={max_abs_diff:.2e}"
        )

    def test_all_ones_prediction_hard_target(self) -> None:
        """Edge case: p = t = 1 everywhere → both losses = 0."""
        t = torch.ones(1, 2, 4, 4, 4)
        dml = dice_semimetric_loss(t, t)
        ref = _soft_dice_reference(t, t)
        assert torch.isclose(dml, ref, atol=1e-6)

    def test_no_overlap_hard(self) -> None:
        """p = 1 where t = 0 and vice versa → both losses = 1."""
        t = torch.zeros(1, 2, 4, 4, 4)
        t[:, :, :2] = 1.0  # foreground in first half
        p = 1.0 - t  # no overlap
        dml = dice_semimetric_loss(p, t)
        ref = _soft_dice_reference(p, t)
        assert torch.isclose(dml, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Acceptance 2: DML properness on soft labels
# ---------------------------------------------------------------------------


class TestDMLSoftLabelProperness:
    """DML is proper on soft labels: symmetric, non-negative, minimal at p=t."""

    def test_dml_is_zero_at_perfect_match(self) -> None:
        """DML(p, t) = 0 when probs == target (soft)."""
        torch.manual_seed(1)
        t = torch.rand(2, 2, 8, 8, 8)  # arbitrary soft targets
        dml = dice_semimetric_loss(t, t)
        assert dml.isfinite(), "DML must be finite at perfect match"
        assert dml.item() >= 0.0, "DML must be non-negative"
        assert dml.item() < 1e-5, f"DML at p=t must be ~0, got {dml.item():.6f}"

    def test_dml_is_nonnegative(self) -> None:
        torch.manual_seed(2)
        p = torch.rand(2, 2, 8, 8, 8)
        t = torch.rand(2, 2, 8, 8, 8)
        dml = dice_semimetric_loss(p, t)
        assert dml.item() >= 0.0

    def test_dml_symmetry_wang2023(self) -> None:
        """Wang-2023 semimetric symmetry: DML(p, t) == DML(t, p).

        Symmetry is the defining property of the Dice semimetric
        (Definition 1 in arXiv:2303.16296).
        """
        torch.manual_seed(3)
        p = torch.rand(2, 2, 8, 8, 8)
        t = torch.rand(2, 2, 8, 8, 8)
        dml_pt = dice_semimetric_loss(p, t)
        dml_tp = dice_semimetric_loss(t, p)
        assert torch.isclose(dml_pt, dml_tp, atol=1e-7), (
            f"DML not symmetric: DML(p,t)={dml_pt.item():.8f}, DML(t,p)={dml_tp.item():.8f}"
        )

    def test_dml_strictly_increasing_under_perturbation(self) -> None:
        """DML is strictly larger under perturbation than at perfect match."""
        torch.manual_seed(4)
        t = torch.rand(2, 2, 8, 8, 8)
        dml_perfect = dice_semimetric_loss(t, t)

        # Small perturbation
        noise_small = 0.05 * torch.randn_like(t)
        p_small = (t + noise_small).clamp(0, 1)
        dml_small = dice_semimetric_loss(p_small, t)

        # Large perturbation
        noise_large = 0.3 * torch.randn_like(t)
        p_large = (t + noise_large).clamp(0, 1)
        dml_large = dice_semimetric_loss(p_large, t)

        assert dml_perfect.item() < dml_small.item(), (
            f"DML not monotone: dml_perfect={dml_perfect.item():.6f} "
            f">= dml_small={dml_small.item():.6f}"
        )
        assert dml_small.item() < dml_large.item(), (
            f"DML not increasing: dml_small={dml_small.item():.6f} "
            f">= dml_large={dml_large.item():.6f}"
        )


# ---------------------------------------------------------------------------
# MONAI DiceLoss improperness demonstration (pins why DML exists)
# ---------------------------------------------------------------------------


class TestMonaiDiceImproperOnSoftLabels:
    """MONAI DiceLoss is NOT minimized at p=t for soft targets.

    MONAI uses sum(p) + sum(t) in the denominator; for soft t the minimum
    is at p=1 (not p=t).  DML (squared denominator) fixes this.
    """

    def test_monai_dice_not_minimized_at_p_equals_t(self) -> None:
        monai = pytest.importorskip("monai")  # noqa: F841
        from monai.losses import DiceLoss as MonaiDiceLoss

        # Soft targets: all 0.5
        t = torch.full((1, 2, 4, 4, 4), 0.5)
        p_match = torch.full((1, 2, 4, 4, 4), 0.5)  # p = t
        p_ones = torch.full((1, 2, 4, 4, 4), 1.0)  # p = 1 everywhere

        monai_dice = MonaiDiceLoss(sigmoid=False, reduction="mean", include_background=True)
        loss_at_match = monai_dice(p_match, t).item()
        loss_at_ones = monai_dice(p_ones, t).item()

        # MONAI Dice: p=1 gives LOWER loss than p=t  → improper
        assert loss_at_ones < loss_at_match, (
            f"Expected MONAI DiceLoss(p=1, t=0.5) < DiceLoss(p=t=0.5), "
            f"got {loss_at_ones:.4f} >= {loss_at_match:.4f}"
        )

        # DML: p=t gives LOWER (minimum) loss → proper
        dml_at_match = dice_semimetric_loss(p_match, t).item()
        dml_at_ones = dice_semimetric_loss(p_ones, t).item()

        assert dml_at_match < dml_at_ones, (
            f"Expected DML(p=t=0.5)=0 < DML(p=1, t=0.5), "
            f"got {dml_at_match:.4f} >= {dml_at_ones:.4f}"
        )

        # Exact improperness numbers for the report
        # MONAI loss at p=t=0.5:  ~0.5
        # MONAI loss at p=1:      ~0.333  → lower, proving improperness
        # DML at p=t=0.5:         ~0.0    → minimum
        # DML at p=1, t=0.5:      ~0.2
        assert abs(dml_at_match) < 1e-5, f"DML must be 0 at p=t, got {dml_at_match:.6f}"


# ---------------------------------------------------------------------------
# Acceptance 3: finite grad, grad ≈ 0 at perfect prediction
# ---------------------------------------------------------------------------


class TestSegmentationLossGradients:
    """SegmentationLoss.forward → scalar; grad finite; grad ≈ 0 at p=t."""

    def _make_cfg(self, **kwargs: object) -> LossConfig:
        defaults = {
            "dice_variant": "dml",
            "ce_variant": "ce",
            "dice_weight": 1.0,
            "ce_weight": 1.0,
            "deep_supervision_weights": (1.0, 0.5, 0.25),
        }
        defaults.update(kwargs)
        return LossConfig(**defaults)  # type: ignore[arg-type]

    def test_finite_grad_arbitrary_prediction(self) -> None:
        torch.manual_seed(5)
        cfg = self._make_cfg()
        loss_fn = SegmentationLoss(cfg)
        logits = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
        target = torch.rand(2, 2, 8, 8, 8)
        loss = loss_fn(logits, target)
        assert loss.isfinite().item(), f"Loss is not finite: {loss.item()}"
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.isfinite().all(), "Gradient contains non-finite values"

    def test_grad_approx_zero_at_perfect_prediction(self) -> None:
        """Gradient w.r.t. logits ≈ 0 when probs == target (DML + CE).

        At p = t both DML and BCE have zero gradient w.r.t. p, so
        d(loss)/d(logits) = d(loss)/d(p) * sigmoid'(logits) = 0.
        """
        torch.manual_seed(6)
        cfg = self._make_cfg()
        loss_fn = SegmentationLoss(cfg)

        # Build logits such that sigmoid(logits) == target exactly
        target = torch.rand(2, 2, 8, 8, 8).clamp(1e-4, 1 - 1e-4)
        logits = torch.logit(target).detach().requires_grad_(True)

        loss = loss_fn(logits, target)
        loss.backward()

        assert logits.grad is not None
        grad_max_abs = logits.grad.abs().max().item()
        # Allow some floating-point tolerance from the clamped logit inversion
        assert grad_max_abs < 1e-3, (
            f"Gradient at perfect prediction should be ≈ 0; max|grad|={grad_max_abs:.2e}"
        )

    def test_focal_ce_finite_grad(self) -> None:
        torch.manual_seed(7)
        cfg = self._make_cfg(ce_variant="focal_ce")
        loss_fn = SegmentationLoss(cfg)
        logits = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
        target = torch.rand(2, 2, 8, 8, 8)
        loss = loss_fn(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.isfinite().all()

    def test_focal_tversky_finite_grad(self) -> None:
        torch.manual_seed(8)
        cfg = self._make_cfg(dice_variant="focal_tversky", ce_variant="focal_ce")
        loss_fn = SegmentationLoss(cfg)
        logits = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
        target = torch.rand(2, 2, 8, 8, 8)
        loss = loss_fn(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.isfinite().all()


# ---------------------------------------------------------------------------
# Acceptance 4: deep supervision weighted sum matches hand computation
# ---------------------------------------------------------------------------


class TestDeepSupervision:
    """Deep supervision: weighted sum of per-head losses."""

    def test_two_head_matches_hand_computation(self) -> None:
        """2-head output → weighted sum must exactly equal the hand computation."""
        torch.manual_seed(9)
        cfg = LossConfig(
            dice_variant="dml",
            ce_variant="ce",
            dice_weight=1.0,
            ce_weight=1.0,
            deep_supervision_weights=(1.0, 0.5),
        )
        loss_fn = SegmentationLoss(cfg)
        gamma = loss_fn._focal_gamma

        b, c, h, w, d = 1, 2, 8, 8, 8
        main_logits = torch.randn(b, c, h, w, d)
        aux_logits = torch.randn(b, c, h // 2, w // 2, d // 2)
        target = torch.rand(b, c, h, w, d)

        # Forward via nn.Module
        loss_module = loss_fn((main_logits, aux_logits), target)

        # Hand computation
        target_down = F.interpolate(target, size=(h // 2, w // 2, d // 2), mode="area")
        loss_main = _compute_single_loss(main_logits, target, cfg, gamma)
        loss_aux = _compute_single_loss(aux_logits, target_down, cfg, gamma)
        expected = 1.0 * loss_main + 0.5 * loss_aux

        assert torch.isclose(loss_module, expected, atol=1e-6), (
            f"Module loss={loss_module.item():.8f} != hand={expected.item():.8f}"
        )

    def test_single_head_no_deep_supervision(self) -> None:
        """Single Tensor output (not a tuple) → no downsampling."""
        cfg = LossConfig()
        loss_fn = SegmentationLoss(cfg)
        logits = torch.randn(1, 2, 8, 8, 8)
        target = torch.rand(1, 2, 8, 8, 8)
        loss = loss_fn(logits, target)
        assert loss.isfinite()

    def test_too_many_heads_raises(self) -> None:
        cfg = LossConfig(deep_supervision_weights=(1.0,))
        loss_fn = SegmentationLoss(cfg)
        head1 = torch.randn(1, 2, 8, 8, 8)
        head2 = torch.randn(1, 2, 8, 8, 8)
        target = torch.rand(1, 2, 8, 8, 8)
        with pytest.raises(SegLossError):
            loss_fn((head1, head2), target)


# ---------------------------------------------------------------------------
# Acceptance 5: Tversky FN-heavy > FP-heavy for alpha < beta
# ---------------------------------------------------------------------------


class TestTverskyFNWeighting:
    """With alpha < beta, FN-heavy preds incur strictly larger Tversky loss."""

    def test_fn_heavy_strictly_greater_than_fp_heavy(self) -> None:
        """FN-heavy pred incurs strictly larger Tversky loss than equal-count FP-heavy.

        alpha=0.3 (FP weight), beta=0.7 (FN weight).

        Construction (equal single-error count):
          target     = [1, 1, 1, 1, 0, 0, 0, 0]  — 4 foreground
          fp_pred    = [1, 1, 1, 1, 1, 0, 0, 0]  — TP=4, FP=1, FN=0  (1 false alarm)
          fn_pred    = [1, 1, 1, 0, 0, 0, 0, 0]  — TP=3, FP=0, FN=1  (1 missed)

        Expected TI values (hand-computed, eps≈0):
          fp_pred: TI = 4 / (4 + 0.3·1 + 0.7·0) = 4/4.3 ≈ 0.9302 → loss ≈ 0.0698
          fn_pred: TI = 3 / (3 + 0.3·0 + 0.7·1) = 3/3.7 ≈ 0.8108 → loss ≈ 0.1892

        Since beta > alpha, the single FN is penalised more than the single FP.
        """
        alpha, beta = 0.3, 0.7
        assert alpha < beta, "Test invariant: alpha < beta"

        target = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 1, 8)

        # FP-heavy: 1 false positive, 0 false negatives
        fp_pred = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 8)
        # FN-heavy: 0 false positives, 1 false negative (equal error count)
        fn_pred = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 1, 8)

        loss_fp = tversky_term(fp_pred, target, alpha=alpha, beta=beta, focal_gamma=None)
        loss_fn = tversky_term(fn_pred, target, alpha=alpha, beta=beta, focal_gamma=None)

        # With alpha=0.3 < beta=0.7: FN weight > FP weight → loss_fn > loss_fp
        assert loss_fn.item() > loss_fp.item(), (
            f"Expected FN-heavy loss ({loss_fn.item():.4f}) > "
            f"FP-heavy loss ({loss_fp.item():.4f}) with alpha={alpha}, beta={beta}"
        )
        # Verify approximate hand-computed values
        assert abs(loss_fp.item() - 0.0698) < 0.01, (
            f"FP loss expected ~0.0698, got {loss_fp.item():.4f}"
        )
        assert abs(loss_fn.item() - 0.1892) < 0.01, (
            f"FN loss expected ~0.1892, got {loss_fn.item():.4f}"
        )

    def test_symmetric_case_alpha_equals_beta(self) -> None:
        """With alpha == beta, the function runs without error (smoke)."""
        alpha = beta = 0.5
        target = torch.tensor([1.0, 1.0, 0.0, 0.0]).reshape(1, 1, 4)
        # fp_pred: tp=0, fp=2, fn=2  |  fn_pred: tp=0, fp=0, fn=2
        fp_pred = torch.tensor([0.0, 0.0, 1.0, 1.0]).reshape(1, 1, 4)
        loss = tversky_term(fp_pred, target, alpha=alpha, beta=beta, focal_gamma=None)
        assert loss.isfinite()


# ---------------------------------------------------------------------------
# Real-mask sanity: DML+CE decreases toward make_soft_targets output
# ---------------------------------------------------------------------------


class TestRealMaskSanity:
    """DML + CE loss decreases as predictions approach a make_soft_targets target."""

    def test_loss_decreases_toward_soft_target(self) -> None:
        """Build a soft target from a synthetic BraTS label; verify loss ordering."""
        label = np.zeros((32, 32, 32), dtype=np.int32)
        # BraTS-2021: ET=4 (enhancing), NETC=1 (necrotic)
        label[8:24, 8:24, 8:24] = 1  # NETC region
        label[12:20, 12:20, 12:20] = 4  # ET region (subset of NETC box)

        cfg_target = TargetConfig(soft=True, sdt_sigma_vox=3.0, tumor_region="tc")
        soft_np = make_soft_targets(label, cfg_target)  # (2, 32, 32, 32)
        target_t = torch.from_numpy(soft_np).unsqueeze(0)  # (1, 2, 32, 32, 32)

        cfg_loss = LossConfig(dice_variant="dml", ce_variant="ce")
        loss_fn = SegmentationLoss(cfg_loss)

        # Far prediction: constant 0.5 everywhere (logits=0)
        logits_far = torch.zeros_like(target_t)
        # Near prediction: logit-invert of the soft target (perfect if no clip)
        logits_near = torch.logit(target_t.clamp(1e-4, 1 - 1e-4))

        loss_far = loss_fn(logits_far, target_t).item()
        loss_near = loss_fn(logits_near, target_t).item()

        assert loss_near < loss_far, (
            f"Loss did not decrease toward target: near={loss_near:.4f}, far={loss_far:.4f}"
        )


# ---------------------------------------------------------------------------
# CE term tests
# ---------------------------------------------------------------------------


class TestCETerm:
    """Unit tests for ce_term (standard and focal)."""

    def test_standard_ce_finite(self) -> None:
        torch.manual_seed(10)
        logits = torch.randn(2, 2, 4, 4, 4)
        target = torch.rand(2, 2, 4, 4, 4)
        loss = ce_term(logits, target, focal_gamma=None)
        assert loss.isfinite()

    def test_focal_ce_finite(self) -> None:
        torch.manual_seed(11)
        logits = torch.randn(2, 2, 4, 4, 4)
        target = torch.rand(2, 2, 4, 4, 4)
        loss = ce_term(logits, target, focal_gamma=2.0)
        assert loss.isfinite()

    def test_focal_ce_downweights_easy_examples(self) -> None:
        """Focal gamma > 0 should reduce loss on confident correct predictions."""
        # Easy correct prediction: logits >> 0 where target ≈ 1
        logits = torch.full((1, 1, 8, 8, 8), 5.0)  # sigmoid ≈ 0.993
        target = torch.ones(1, 1, 8, 8, 8)
        ce = ce_term(logits, target, focal_gamma=None)
        focal = ce_term(logits, target, focal_gamma=2.0)
        # Focal should be lower (down-weighted) for easy examples
        assert focal.item() < ce.item(), (
            f"Focal CE ({focal.item():.4f}) should be < CE ({ce.item():.4f}) on easy examples"
        )

    def test_shape_mismatch_raises(self) -> None:

        logits = torch.randn(2, 2, 4)
        target = torch.rand(2, 2, 5)
        with pytest.raises(SegLossError):
            ce_term(logits, target, focal_gamma=None)


# ---------------------------------------------------------------------------
# DML edge / error cases
# ---------------------------------------------------------------------------


class TestDMLErrors:
    def test_shape_mismatch_raises(self) -> None:

        p = torch.rand(2, 2, 4)
        t = torch.rand(2, 2, 5)
        with pytest.raises(SegLossError):
            dice_semimetric_loss(p, t)

    def test_bad_reduction_raises(self) -> None:

        p = torch.rand(1, 2, 4)
        with pytest.raises(SegLossError):
            dice_semimetric_loss(p, p, reduction="invalid")

    def test_1d_raises(self) -> None:

        p = torch.rand(8)
        with pytest.raises(SegLossError):
            dice_semimetric_loss(p, p)

    def test_reduction_none_returns_per_bc(self) -> None:
        n_batch, n_chan = 3, 2
        p = torch.rand(n_batch, n_chan, 4, 4, 4)
        out = dice_semimetric_loss(p, p, reduction="none")
        assert out.shape == (n_batch, n_chan)
        # At p=t, all values should be 0
        assert out.abs().max().item() < 1e-5

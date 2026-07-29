"""Unit tests for CFMLoss pseudo-Huber (B9).

Verifies that the ``'huber'`` norm branch in :class:`CFMLoss` satisfies the
expected mathematical properties of the pseudo-Huber loss (Song & Dhariwal
ICLR 2024) and produces numerically different results from L1 and L2.
"""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.losses.base import LossInputs
from vena.model.fm.controlnet.losses.cfm import CFMLoss, _pseudo_huber

pytestmark = pytest.mark.unit


def _make_inputs(pred: torch.Tensor, target: torch.Tensor) -> LossInputs:
    """Build minimal LossInputs for the CFM-only path."""
    B = pred.shape[0]  # noqa: N806 — batch-size by ML convention
    return LossInputs(
        x_clean=target,
        noise=torch.zeros_like(target),
        x_t=target,
        timesteps=torch.zeros(B, dtype=torch.long),
        u_target=target,
        v_orig=pred,
    )


class TestPseudoHuberHelper:
    """Tests for the module-level ``_pseudo_huber`` function."""

    def test_zero_residual_gives_zero(self) -> None:
        t = torch.zeros(3, 4, 5, 6, 7)
        out = _pseudo_huber(t, t, delta=0.90)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_elementwise_shape_preserved(self) -> None:
        pred = torch.randn(2, 4, 8, 8, 8)
        target = torch.randn_like(pred)
        out = _pseudo_huber(pred, target, delta=0.90)
        assert out.shape == pred.shape

    def test_quadratic_near_zero(self) -> None:
        """For small |r| << delta the loss should be ≈ r²/2 (L2-like)."""
        delta = 1.0
        r = 0.01
        pred = torch.tensor([r])
        target = torch.zeros(1)
        huber = _pseudo_huber(pred, target, delta=delta).item()
        expected = r**2 / 2  # leading quadratic term
        assert abs(huber - expected) < 1e-5

    def test_linear_growth_for_large_residuals(self) -> None:
        """For |r| >> delta the loss grows linearly (L1-like asymptote)."""
        delta = 0.5
        r_small = torch.tensor([10.0])
        r_big = torch.tensor([20.0])
        target = torch.zeros(1)
        l_small = _pseudo_huber(r_small, target, delta=delta).item()
        l_big = _pseudo_huber(r_big, target, delta=delta).item()
        # Linear regime: doubling |r| should roughly double the loss.
        ratio = l_big / l_small
        assert 1.8 < ratio < 2.2

    def test_nonnegative(self) -> None:
        pred = torch.randn(4, 4, 4, 4, 4)
        target = torch.randn_like(pred)
        out = _pseudo_huber(pred, target, delta=0.90)
        assert (out >= 0).all()


class TestCFMLossHuberNorm:
    """Tests for ``CFMLoss(norm='huber')`` integration."""

    def test_invalid_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="norm must be"):
            CFMLoss(norm="invalid_norm")

    def test_huber_mean_reduction(self) -> None:
        loss_fn = CFMLoss(norm="huber", reduction="mean", delta=0.90)
        pred = torch.randn(2, 4, 8, 8, 8)
        target = torch.randn_like(pred)
        inp = _make_inputs(pred, target)
        out = loss_fn(inp)
        assert out.ndim == 0  # scalar
        assert out.item() > 0

    def test_huber_sum_reduction(self) -> None:
        loss_fn = CFMLoss(norm="huber", reduction="sum", delta=0.90)
        pred = torch.randn(2, 4, 4, 4, 4)
        target = torch.randn_like(pred)
        inp = _make_inputs(pred, target)
        out = loss_fn(inp)
        assert out.ndim == 0

    def test_huber_none_reduction(self) -> None:
        loss_fn = CFMLoss(norm="huber", reduction="none", delta=0.90)
        pred = torch.randn(2, 4, 4, 4, 4)
        target = torch.randn_like(pred)
        inp = _make_inputs(pred, target)
        out = loss_fn(inp)
        assert out.shape == pred.shape

    def test_huber_differs_from_l1_and_l2(self) -> None:
        """With the same inputs, Huber, L1, and L2 should give different values."""
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 8, 8, 8)
        target = torch.randn_like(pred)
        inp = _make_inputs(pred, target)

        l_l1 = CFMLoss(norm="l1", reduction="mean")(inp).item()
        l_l2 = CFMLoss(norm="l2", reduction="mean")(inp).item()
        l_huber = CFMLoss(norm="huber", reduction="mean", delta=0.90)(inp).item()

        assert l_l1 != pytest.approx(l_huber, rel=1e-4)
        assert l_l2 != pytest.approx(l_huber, rel=1e-4)

    def test_huber_delta_stored(self) -> None:
        loss_fn = CFMLoss(norm="huber", delta=1.23)
        assert loss_fn.delta == pytest.approx(1.23)

    def test_huber_zero_residual_zero_loss(self) -> None:
        loss_fn = CFMLoss(norm="huber", reduction="mean")
        pred = torch.ones(2, 4, 4, 4, 4)
        inp = _make_inputs(pred, pred)
        out = loss_fn(inp)
        assert out.item() == pytest.approx(0.0, abs=1e-7)

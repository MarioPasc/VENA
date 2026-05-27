"""Unit tests for CFM loss + CompositeLoss + builder."""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.losses import (
    CFMLoss,
    CompositeLoss,
    LossInputs,
    build_loss,
)


def _make_inputs(B: int = 1, C: int = 4, h: int = 4, w: int = 4, d: int = 4) -> LossInputs:
    x1 = torch.randn(B, C, h, w, d)
    x0 = torch.randn(B, C, h, w, d)
    x_t = 0.5 * x1 + 0.5 * x0
    u = x1 - x0
    v = torch.randn(B, C, h, w, d)
    return LossInputs(
        x_clean=x1, noise=x0, x_t=x_t, timesteps=torch.zeros(B, dtype=torch.long),
        u_target=u, v_orig=v,
    )


@pytest.mark.unit
def test_cfm_loss_finite() -> None:
    loss = CFMLoss()
    inputs = _make_inputs()
    val = loss(inputs)
    assert torch.isfinite(val)
    assert val.ndim == 0


@pytest.mark.unit
def test_cfm_loss_zero_when_prediction_is_target() -> None:
    loss = CFMLoss()
    inputs = _make_inputs()
    inputs = LossInputs(
        x_clean=inputs.x_clean, noise=inputs.noise, x_t=inputs.x_t,
        timesteps=inputs.timesteps, u_target=inputs.u_target,
        v_orig=inputs.u_target,  # perfect prediction
    )
    val = loss(inputs)
    assert torch.allclose(val, torch.zeros(()), atol=1e-6)


@pytest.mark.unit
def test_build_loss_s1_returns_cfm_only() -> None:
    composite = build_loss("S1", {"cfm": {"weight": 1.0}})
    assert isinstance(composite, CompositeLoss)
    assert set(composite.terms.keys()) == {"cfm"}
    assert composite.requires_perturbed_pass is False
    assert composite.stage == "S1"


@pytest.mark.unit
def test_build_loss_s2_includes_contrastive_and_requires_perturb() -> None:
    composite = build_loss("S2", {})
    assert set(composite.terms.keys()) == {"cfm", "contrastive"}
    assert composite.requires_perturbed_pass is True


@pytest.mark.unit
def test_build_loss_s3_adds_reconstruction() -> None:
    composite = build_loss("S3", {})
    assert set(composite.terms.keys()) == {"cfm", "contrastive", "reconstruction"}
    assert composite.requires_perturbed_pass is True


@pytest.mark.unit
def test_build_loss_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="unknown curriculum stage"):
        build_loss("S99", {})


@pytest.mark.unit
def test_s2_contrastive_stub_raises_on_forward() -> None:
    composite = build_loss("S2", {})
    inputs = _make_inputs()
    with pytest.raises(NotImplementedError, match="S2 commit"):
        composite(inputs)


@pytest.mark.unit
def test_composite_returns_total_and_per_term() -> None:
    composite = build_loss("S1", {"cfm": {"weight": 0.5}})
    inputs = _make_inputs()
    total, per_term = composite(inputs)
    assert total.ndim == 0
    assert "cfm" in per_term and "total" in per_term
    assert torch.allclose(total.detach(), per_term["total"])

"""Unit tests for the composite-loss weight schedule (proposal §3 anneal)."""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.controlnet.losses import LossInputs, build_loss
from vena.model.fm.controlnet.losses.schedule import (
    StaticWeight,
    StepHalfWeight,
    build_schedule,
)

pytestmark = pytest.mark.unit


def test_static_weight_independent_of_step() -> None:
    w = StaticWeight(0.5)
    assert w.at(0, 1000) == 0.5
    assert w.at(999, 1000) == 0.5
    assert w.at(None, None) == 0.5


def test_step_half_weight_anneals_at_half() -> None:
    w = StepHalfWeight(w0=0.01, factor=0.1)
    assert w.at(0, 1000) == pytest.approx(0.01)
    assert w.at(499, 1000) == pytest.approx(0.01)
    assert w.at(500, 1000) == pytest.approx(0.001)
    assert w.at(999, 1000) == pytest.approx(0.001)


def test_step_half_falls_back_to_w0_without_total() -> None:
    w = StepHalfWeight(w0=0.01, factor=0.1)
    # Unknown total ⇒ schedule cannot anneal; return w0.
    assert w.at(100, None) == pytest.approx(0.01)
    assert w.at(None, 1000) == pytest.approx(0.01)


def test_build_schedule_static_default() -> None:
    sched = build_schedule(1.0, None)
    assert isinstance(sched, StaticWeight)
    assert sched.at(123, 456) == 1.0


def test_build_schedule_step_half() -> None:
    sched = build_schedule(0.01, {"kind": "step_half", "factor": 0.1})
    assert isinstance(sched, StepHalfWeight)
    assert sched.at(0, 1000) == pytest.approx(0.01)
    assert sched.at(800, 1000) == pytest.approx(0.001)


def test_build_schedule_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown schedule.kind"):
        build_schedule(1.0, {"kind": "cosine"})


def _toy_inputs(B: int = 1) -> LossInputs:
    v = torch.randn(B, 4, 4, 4, 4)
    return LossInputs(
        x_clean=v,
        noise=v,
        x_t=v,
        timesteps=torch.zeros(B, dtype=torch.long),
        u_target=v,
        v_orig=v,
        v_perturb=torch.zeros_like(v),
        m_wt=torch.ones(B, 1, 4, 4, 4),
        m_bg=torch.zeros(B, 1, 4, 4, 4),
    )


def test_composite_applies_schedule_for_contrastive() -> None:
    cfg = {
        "cfm": {"weight": 1.0},
        "contrastive": {
            "weight": 0.01,
            "schedule": {"kind": "step_half", "factor": 0.1},
        },
    }
    composite = build_loss("S2", cfg)
    inputs = _toy_inputs()

    _, pre = composite(inputs, global_step=0, total_steps=1000)
    _, post = composite(inputs, global_step=800, total_steps=1000)

    assert pre["contrastive_weight"].item() == pytest.approx(0.01)
    assert post["contrastive_weight"].item() == pytest.approx(0.001)
    # cfm weight is static at 1.0 in both phases.
    assert pre["cfm_weight"].item() == pytest.approx(1.0)
    assert post["cfm_weight"].item() == pytest.approx(1.0)


def test_composite_no_step_falls_back_to_w0() -> None:
    """Calling composite(...) without global_step keeps the legacy contract."""
    cfg = {
        "cfm": {"weight": 1.0},
        "contrastive": {"weight": 0.01, "schedule": {"kind": "step_half"}},
    }
    composite = build_loss("S2", cfg)
    inputs = _toy_inputs()
    _, per_term = composite(inputs)  # no step args
    assert per_term["contrastive_weight"].item() == pytest.approx(0.01)

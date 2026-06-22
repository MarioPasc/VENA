"""Unit tests for :class:`OutputScaleRampCallback`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vena.model.fm.lightning.callbacks import OutputScaleRampCallback


@pytest.mark.unit
def test_ramp_value_at_step_0_is_near_zero() -> None:
    cb = OutputScaleRampCallback(ramp_steps=5000, steepness=10.0)
    # sigmoid(-5) ≈ 0.0067 with steepness=10 at progress=0
    assert cb.ramp_value(0) == pytest.approx(0.006693, abs=1e-4)


@pytest.mark.unit
def test_ramp_value_at_midpoint_is_half() -> None:
    cb = OutputScaleRampCallback(ramp_steps=5000, steepness=10.0)
    assert cb.ramp_value(2500) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.unit
def test_ramp_value_at_ramp_steps_clamps_to_one() -> None:
    cb = OutputScaleRampCallback(ramp_steps=5000, steepness=10.0)
    assert cb.ramp_value(5000) == 1.0
    assert cb.ramp_value(50_000) == 1.0


@pytest.mark.unit
def test_ramp_value_is_monotonically_increasing() -> None:
    cb = OutputScaleRampCallback(ramp_steps=1000, steepness=10.0)
    prev = cb.ramp_value(0)
    for step in range(50, 1000, 50):
        cur = cb.ramp_value(step)
        assert cur > prev, f"non-monotone at step {step}: {cur} vs prev {prev}"
        prev = cur


@pytest.mark.unit
def test_ramp_steepness_makes_transition_sharper() -> None:
    # Sharper steepness → faster transition through the midpoint.
    cb_soft = OutputScaleRampCallback(ramp_steps=1000, steepness=4.0)
    cb_sharp = OutputScaleRampCallback(ramp_steps=1000, steepness=20.0)
    # At quarter-progress, sharper ramp must be closer to 0; at 3/4 closer to 1.
    assert cb_sharp.ramp_value(250) < cb_soft.ramp_value(250)
    assert cb_sharp.ramp_value(750) > cb_soft.ramp_value(750)


@pytest.mark.unit
def test_ramp_callback_writes_buffer() -> None:
    """The callback fills ``pl_module.controlnet.output_scale`` from ramp_value(global_step)."""
    cb = OutputScaleRampCallback(ramp_steps=1000, steepness=10.0)

    # Minimal mocks. The callback only touches ``trainer.global_step`` and
    # ``pl_module.controlnet.output_scale``.
    output_scale = torch.tensor(1.0)
    controlnet = SimpleNamespace(output_scale=output_scale)
    pl_module = SimpleNamespace(controlnet=controlnet)
    trainer = SimpleNamespace(global_step=0)

    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    assert output_scale.item() == pytest.approx(cb.ramp_value(0))

    trainer.global_step = 500
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    assert output_scale.item() == pytest.approx(0.5, abs=1e-6)

    trainer.global_step = 2000  # past ramp_steps → clamped
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    assert output_scale.item() == 1.0


@pytest.mark.unit
def test_ramp_callback_noop_without_controlnet() -> None:
    """If the module has no ``controlnet`` attr, the callback is a silent no-op."""
    cb = OutputScaleRampCallback()
    pl_module = SimpleNamespace()  # no .controlnet
    trainer = SimpleNamespace(global_step=0)
    # Must not raise.
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)


@pytest.mark.unit
def test_ramp_callback_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="ramp_steps must be positive"):
        OutputScaleRampCallback(ramp_steps=0)
    with pytest.raises(ValueError, match="ramp_steps must be positive"):
        OutputScaleRampCallback(ramp_steps=-100)

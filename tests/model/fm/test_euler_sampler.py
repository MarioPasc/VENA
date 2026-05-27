"""Unit tests for EulerSampler."""

from __future__ import annotations

import pytest
import torch

from vena.model.fm.inference import EulerSampler
from vena.model.fm.sampler.rflow import RFlowEngine


@pytest.mark.unit
@pytest.mark.parametrize("nfe", [1, 2, 5])
def test_euler_output_shape(nfe: int) -> None:
    engine = RFlowEngine(num_train_timesteps=100, use_discrete_timesteps=True)
    sampler = EulerSampler(scheduler=engine.scheduler)

    def model_call(x, t):
        return torch.zeros_like(x)

    x0 = torch.randn(2, 4, 8, 8, 8)
    out = sampler.sample(model_call, x0, num_inference_steps=nfe)
    assert out.shape == x0.shape


@pytest.mark.unit
def test_euler_zero_velocity_preserves_input() -> None:
    """With v ≡ 0 the integrator must return x0 unchanged."""
    engine = RFlowEngine(num_train_timesteps=100, use_discrete_timesteps=True)
    sampler = EulerSampler(scheduler=engine.scheduler)

    def model_call(x, t):
        return torch.zeros_like(x)

    x0 = torch.randn(1, 4, 8, 8, 8)
    out = sampler.sample(model_call, x0, num_inference_steps=5)
    assert torch.allclose(out, x0)

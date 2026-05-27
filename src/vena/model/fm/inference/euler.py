"""Euler integration over MONAI's rectified-flow scheduler.

Per :class:`monai.networks.schedulers.rectified_flow.RFlowScheduler.step`,
each call to ``step(v, t, x)`` performs the Euler update

.. math::
    x_{t-\\Delta t} \\;=\\; x_t \\;+\\; v(x_t, t) \\cdot \\Delta t,

with :math:`\\Delta t = 1 / N` for ``N`` inference steps. This implementation
just iterates the scheduler-provided timestep schedule and forwards model
output through ``scheduler.step``. NFE equals ``num_inference_steps``.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import BaseSampler, SamplerCallable


class EulerSampler(BaseSampler):
    """Euler integration on top of a MONAI :class:`RFlowScheduler`.

    Parameters
    ----------
    scheduler : RFlowScheduler
        The scheduler whose ``set_timesteps`` + ``step`` we drive. Typically
        sourced from :attr:`vena.model.fm.sampler.rflow.RFlowEngine.scheduler`.
    input_img_size_numel : int | None
        Forwarded to ``scheduler.set_timesteps`` for resolution-aware timestep
        transforms. ``None`` uses the scheduler's default.
    """

    def __init__(
        self,
        scheduler: Any,
        input_img_size_numel: int | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.input_img_size_numel = input_img_size_numel

    @torch.inference_mode()
    def sample(
        self,
        model_call: SamplerCallable,
        x0: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        device = x0.device
        kwargs: dict[str, Any] = {"num_inference_steps": int(num_inference_steps), "device": device}
        if self.input_img_size_numel is not None:
            kwargs["input_img_size_numel"] = int(self.input_img_size_numel)
        self.scheduler.set_timesteps(**kwargs)

        x = x0.clone()
        for t in self.scheduler.timesteps:
            t_int = int(t.item()) if torch.is_tensor(t) else int(t)
            t_batch = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
            v = model_call(x, t_batch)
            x, _ = self.scheduler.step(model_output=v, timestep=t_int, sample=x)
        return x

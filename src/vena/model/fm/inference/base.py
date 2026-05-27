"""Sampler ABC for rectified-flow inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

#: A model callable signature ``v = model_call(x_t, timestep)``. Takes the noisy
#: latent ``(B, C, h, w, d)`` and a ``(B,)`` integer-timestep tensor; returns the
#: predicted velocity at the same shape.
SamplerCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseSampler(ABC):
    """Sample a clean latent by integrating a velocity field.

    Concrete samplers expose a single :meth:`sample` method. The number of
    function evaluations (NFE) is the user-facing budget knob; each sub-class
    decides how many model calls per ODE step (Euler: 1, Heun: 2).
    """

    @abstractmethod
    def sample(
        self,
        model_call: SamplerCallable,
        x0: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Integrate from noise ``x0`` to clean latent.

        Parameters
        ----------
        model_call : callable
            ``v = model_call(x_t, timestep)`` returning the velocity prediction.
        x0 : Tensor
            Noise sample ``(B, C, h, w, d)``.
        num_inference_steps : int
            Number of sampler steps. For an Euler sampler, NFE = num_inference_steps.

        Returns
        -------
        Tensor
            The predicted clean latent ``x1``.
        """

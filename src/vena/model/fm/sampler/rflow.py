"""Rectified-flow noising and target-velocity primitives.

We thinly wrap :class:`monai.networks.schedulers.rectified_flow.RFlowScheduler`
to expose the three operations the trainer needs:

* :meth:`RFlowEngine.sample_timesteps` — sample integer ``t ~ U{0, T-1}``.
* :meth:`RFlowEngine.add_noise` — produce ``x_t`` on the straight-line
  interpolant between clean latent ``x1`` and Gaussian noise ``x0``.
* :meth:`RFlowEngine.target_velocity` — ``u_t = x1 - x0`` (rectified flow).

MONAI's ``add_noise(original_samples=x1, noise=x0, timesteps=t)`` maps the
integer timestep ``t`` to ``t̄ = 1 - t/T``, giving
``x_t = t̄ * x1 + (1 - t̄) * x0``. For low ``t`` (close to 0) this collapses to
clean ``x1``; for high ``t`` (close to ``T``) it collapses to pure noise ``x0``.
The MAISI training script uses ``v_target = x1 - x0`` consistently; we follow
the same convention.

Reference: Liu, Gong & Liu *Flow Straight and Fast* (ICLR 2023, arXiv:2209.03003).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class RFlowEngine:
    """Thin wrapper around MONAI's :class:`RFlowScheduler`.

    Parameters
    ----------
    num_train_timesteps : int
        Number of discrete training timesteps. The MAISI checkpoint stores
        this as ``1000``; we keep the same value by default to stay aligned
        with the warm-started weights.
    use_discrete_timesteps : bool
        Whether to round sampled timesteps to integer codes. The MAISI trunk
        consumes integer timesteps via its sinusoidal embedding, so we keep
        ``True``.
    sample_method : str
        ``"uniform"`` or ``"logit-normal"``. The MAISI v2 paper §3.2 reports
        improvements from logit-normal at high resolution; we keep
        ``"uniform"`` for S1 to stay closest to the trunk's pretraining
        distribution.
    """

    num_train_timesteps: int = 1000
    use_discrete_timesteps: bool = True
    sample_method: str = "uniform"

    def __post_init__(self) -> None:
        from monai.networks.schedulers.rectified_flow import RFlowScheduler

        self._scheduler = RFlowScheduler(
            num_train_timesteps=self.num_train_timesteps,
            use_discrete_timesteps=self.use_discrete_timesteps,
            sample_method=self.sample_method,
        )

    @property
    def scheduler(self) -> RFlowScheduler:
        """Underlying MONAI scheduler (for inference samplers later)."""
        return self._scheduler

    def sample_timesteps(self, x_clean: torch.Tensor) -> torch.Tensor:
        """Sample ``(B,)`` integer timesteps on the same device as ``x_clean``.

        Uses ``RFlowScheduler.sample_timesteps`` which respects the configured
        ``sample_method`` and any resolution-aware timestep transform.
        """
        return self._scheduler.sample_timesteps(x_clean)

    def add_noise(
        self, x_clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Compute the rectified-flow interpolant ``x_t``.

        ``x_t = (1 - t/T) * x_clean + (t/T) * noise`` (continuous), discretised
        through the integer ``timesteps`` codes.
        """
        return self._scheduler.add_noise(original_samples=x_clean, noise=noise, timesteps=timesteps)

    @staticmethod
    def target_velocity(x_clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Return ``u_t = x_clean - noise`` (rectified-flow target).

        This is shape-broadcastable and independent of ``t``.
        """
        return x_clean - noise

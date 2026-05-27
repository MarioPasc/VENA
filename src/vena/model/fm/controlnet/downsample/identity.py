"""Identity downsampler — passes the tensor through unchanged.

Used when the input already arrives at latent resolution, as is the case for
``masks/tumor_latent`` in ``UCSFPDGM_latents.h5`` (pre-computed at write time
by the latent-H5 converter).
"""

from __future__ import annotations

import torch

from .base import AbstractDownsampler


class IdentityDownsampler(AbstractDownsampler):
    """No-op downsampler."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

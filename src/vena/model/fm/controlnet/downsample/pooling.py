"""Strided average-pooling downsampler.

Matches the per-class average pooling used by
``vena.data.h5.ucsf_pdgm.latent_domain.convert.UCSFPDGMLatentH5Converter`` to
write ``masks/tumor_latent`` to the latents H5. Use this when an ablation
needs to re-derive a tumour-mask channel on the fly from an image-resolution
soft segmentation, matching the H5's pre-baked statistics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AbstractDownsampler


class AvgPoolDownsampler(AbstractDownsampler):
    """Strided 3-D average pooling by an integer factor."""

    def __init__(self, factor: int = 4) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError(f"factor must be >= 1; got {factor}")
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, kernel_size=self.factor, stride=self.factor)

"""Nearest-neighbour downsampler with optional binarisation.

Matches proposal §3.3: ``M_WT downsampled by nearest-neighbour, binarised``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AbstractDownsampler


class NearestDownsampler(AbstractDownsampler):
    """Nearest-neighbour downsampling by an integer factor.

    Parameters
    ----------
    factor : int
        Stride per spatial axis. Output shape is ``input // factor`` along
        each of H, W, D.
    binarise_threshold : float | None
        If given, applies ``(x > threshold).float()`` to the resampled output.
        For VENA the tumour mask is a binary mask, so ``0.5`` is the natural
        default if this operator is fed a soft union.
    """

    def __init__(self, factor: int = 4, binarise_threshold: float | None = None) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError(f"factor must be >= 1; got {factor}")
        self.factor = int(factor)
        self.binarise_threshold = binarise_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(x, scale_factor=1.0 / self.factor, mode="nearest")
        if self.binarise_threshold is not None:
            y = (y > self.binarise_threshold).to(x.dtype)
        return y

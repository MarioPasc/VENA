"""Trilinear downsampler for soft maps.

Appropriate for soft priors (Frangi vesselness, perfusion scores, susceptibility
maps) where preserving graded values matters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AbstractDownsampler


class TrilinearDownsampler(AbstractDownsampler):
    """Trilinear downsampling by an integer factor."""

    def __init__(self, factor: int = 4, align_corners: bool = False) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError(f"factor must be >= 1; got {factor}")
        self.factor = int(factor)
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=1.0 / self.factor,
            mode="trilinear",
            align_corners=self.align_corners,
        )

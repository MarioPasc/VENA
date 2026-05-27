"""Abstract base class for image-to-latent downsamplers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractDownsampler(nn.Module, ABC):
    """Map an image-space tensor to latent-space spatial resolution.

    Input  : ``(B, C, H, W, D)``.
    Output : ``(B, C, H/f, W/f, D/f)`` (or a user-specified target shape).

    Subclasses are :class:`torch.nn.Module` so they can hold learnable
    parameters if needed (a future learned downsampler). Stateless operators
    register no parameters; their ``forward`` is pure.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the spatial downsampling."""

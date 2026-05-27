"""Capped L^p background-reconstruction loss (proposal §5.4).

Lands in the S3 commit. Stub for now.
"""

from __future__ import annotations

import torch

from .base import AbstractFMLoss, LossInputs


class CappedLpReconLoss(AbstractFMLoss):
    """Stub for proposal §5.4 — implemented in the S3 follow-up commit."""

    def __init__(self, p: int = 4, delta: float = 2.0) -> None:
        super().__init__()
        self.p = int(p)
        self.delta = float(delta)

    def forward(self, inputs: LossInputs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "CappedLpReconLoss lands in the S3 commit (proposal §5.4). "
            "S1 should not request it via build_loss."
        )

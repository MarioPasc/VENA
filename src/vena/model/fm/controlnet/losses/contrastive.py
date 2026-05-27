"""Tumour-downweighted contrastive regularisation (proposal §5.3).

Lands in the S2 commit. The current file ships as a stub so the builder can
already wire it up — calling it raises ``NotImplementedError`` with a precise
pointer to the proposal section.
"""

from __future__ import annotations

import torch

from .base import AbstractFMLoss, LossInputs


class ContrastiveTumourLoss(AbstractFMLoss):
    """Stub for proposal §5.3 — implemented in the S2 follow-up commit."""

    def __init__(
        self,
        lambda_roi: float = 0.3,
        lambda_bg: float = 1.0,
        delta: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_roi = float(lambda_roi)
        self.lambda_bg = float(lambda_bg)
        self.delta = float(delta)

    def forward(self, inputs: LossInputs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "ContrastiveTumourLoss lands in the S2 commit (proposal §5.3). "
            "S1 should not request it via build_loss."
        )

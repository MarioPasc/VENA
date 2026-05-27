"""Conditional flow-matching loss (rectified flow, MSE on velocity)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AbstractFMLoss, LossInputs


class CFMLoss(AbstractFMLoss):
    r"""Rectified-flow regression loss on the velocity field.

    :math:`\mathcal{L}_\text{CFM} = \mathbb{E}\left[ \| G_\theta(x_t, t, c) -
    (x_1 - x_0) \|_2^2 \right]`

    Per the MAISI-v2 reference implementation we keep the MSE formulation
    (proposal §5.2). The MAISI training script actually uses an L1 default;
    we follow the proposal's text rather than the upstream script.

    Parameters
    ----------
    reduction : {"mean", "sum"}
        ``"mean"`` reduces over batch and spatial dims (default); ``"sum"``
        sums (useful only for testing scale invariants).
    norm : {"l2", "l1"}
        ``"l2"`` is the proposal default. ``"l1"`` is provided for an easy
        switch should MAISI's L1 prove more stable in practice.
    """

    def __init__(self, reduction: str = "mean", norm: str = "l2") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")
        if norm not in ("l2", "l1"):
            raise ValueError(f"norm must be 'l2' or 'l1'; got {norm!r}")
        self.reduction = reduction
        self.norm = norm

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        if self.norm == "l2":
            return F.mse_loss(inputs.v_orig, inputs.u_target, reduction=self.reduction)
        return F.l1_loss(inputs.v_orig, inputs.u_target, reduction=self.reduction)

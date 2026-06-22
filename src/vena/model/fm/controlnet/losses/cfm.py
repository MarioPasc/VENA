"""Conditional flow-matching loss (rectified flow, MSE on velocity).

S1 v3 (2026-06-22) adds optional region-weighted reduction. With
``reduction="none"`` and a :class:`RegionWeights` config, the loss applies
disjoint per-region weights to the per-voxel L1 / L2 tensor before reducing
— addressing the 0.095 %-of-loss-in-WT imbalance documented in
``.claude/notes/review/2026-06-22_s1_v2_tumor_synthesis_failure_diagnosis.md``
§3.1. ``RegionWeights(enabled=False)`` (or omission) is byte-identical to
the legacy mean-reduction path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AbstractFMLoss, LossInputs
from .region_weights import RegionWeights, build_region_weight_tensor


class CFMLoss(AbstractFMLoss):
    r"""Rectified-flow regression loss on the velocity field.

    :math:`\mathcal{L}_\text{CFM} = \mathbb{E}\left[ \| G_\theta(x_t, t, c) -
    (x_1 - x_0) \|_2^2 \right]`

    Per the MAISI-v2 reference implementation we keep the MSE formulation
    (proposal §5.2). The MAISI training script actually uses an L1 default;
    we follow the proposal's text rather than the upstream script.

    Parameters
    ----------
    reduction : {"none", "mean", "sum"}
        ``"mean"`` reduces over batch and spatial dims (S1 v2 default);
        ``"sum"`` sums (useful only for testing scale invariants);
        ``"none"`` is required when ``region_weights`` is supplied — the
        per-voxel loss is multiplied by the region-weight tensor and reduced
        as ``(loss * w).sum() / w.sum()``.
    norm : {"l2", "l1"}
        ``"l1"`` matches T1C-RFlow and the S1 v2 baseline; ``"l2"`` is the
        proposal default and is kept for back-compat.
    region_weights : RegionWeights | None
        When non-None and ``enabled=True``, the loss is region-weighted (S1
        v3). When None, behaviour is identical to S1 v2.

    Raises
    ------
    ValueError
        If ``reduction != "none"`` while a non-None ``region_weights`` has
        ``enabled=True`` — the two are mutually exclusive (region weighting
        requires the per-voxel tensor).
    """

    def __init__(
        self,
        reduction: str = "mean",
        norm: str = "l2",
        region_weights: RegionWeights | None = None,
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum'; got {reduction!r}")
        if norm not in ("l2", "l1"):
            raise ValueError(f"norm must be 'l2' or 'l1'; got {norm!r}")
        if region_weights is not None and region_weights.enabled and reduction != "none":
            raise ValueError(
                f"region_weights.enabled=True requires reduction='none'; got reduction={reduction!r}"
            )
        self.reduction = reduction
        self.norm = norm
        self.region_weights = region_weights

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        # Region-weighted path (S1 v3 default).
        if self.region_weights is not None and self.region_weights.enabled:
            if self.norm == "l2":
                voxel = F.mse_loss(inputs.v_orig, inputs.u_target, reduction="none")
            else:
                voxel = F.l1_loss(inputs.v_orig, inputs.u_target, reduction="none")
            w = build_region_weight_tensor(
                self.region_weights,
                inputs.m_brain,
                inputs.m_tumor,
                channels=voxel.shape[1],
            )
            assert w is not None, "build_region_weight_tensor returned None despite enabled=True"
            # (loss * w).sum() / w.sum() — clamp the denominator only at
            # floor-numerical level so a degenerate all-zero-weight tensor
            # surfaces as a non-finite loss rather than a silent 0.
            return (voxel * w).sum() / w.sum().clamp_min(1e-12)
        # Legacy mean/sum path.
        if self.norm == "l2":
            return F.mse_loss(inputs.v_orig, inputs.u_target, reduction=self.reduction)
        return F.l1_loss(inputs.v_orig, inputs.u_target, reduction=self.reduction)

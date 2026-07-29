"""Conditional flow-matching loss (rectified flow, velocity regression).

Production default is ``norm='l1'`` (all runs since S1 v2, 2026-06-20).
Constructor default remains ``norm='l2'`` for backward compatibility with
unit tests that do not set it explicitly; every production YAML sets norm
explicitly. The ``'huber'`` option (pseudo-Huber, Song & Dhariwal ICLR
2024) is available for the §18 ablation arm C; δ is set from the L1 arm's
median absolute velocity residual (≈ 0.90).

S1 v3 (2026-06-22) adds optional region-weighted reduction.  Two modes:

* **Sub-region mode** (:class:`RegionWeights`) — five regions (BG,
  Brain-not-WT, NETC, ED, ET).  With ``reduction="none"`` and an enabled
  config the loss applies disjoint per-region weights, addressing the
  0.095 %-of-loss-in-WT imbalance documented in
  ``.claude/notes/review/2026-06-22_s1_v2_tumor_synthesis_failure_diagnosis.md``
  §3.1.

* **Brain/TC mode** (:class:`BrainTCWeights`) — three strict-partition
  regions (BG, Brain, TC).  Default ``{brain: 1.0, tc: 1.0}`` is
  numerically identical to ``F.l1_loss(reduction="mean")``.  Activated when
  the YAML ``loss.cfm.region_weights`` block carries ``brain``/``tc`` keys.

``RegionWeights(enabled=False)`` (or omission) is byte-identical to the
legacy mean-reduction path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 — standard PyTorch alias

from .base import AbstractFMLoss, LossInputs
from .region_weights import (
    BrainTCWeights,
    RegionWeights,
    build_brain_tc_weight_tensor,
    build_region_weight_tensor,
)


def _pseudo_huber(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    """Pseudo-Huber loss, elementwise (Song & Dhariwal ICLR 2024).

    Quadratic for ``|r| < delta``, approximately linear beyond.  Numerically
    stable via the sqrt form: ``δ²(√(1+(r/δ)²) - 1)``.

    Parameters
    ----------
    pred : torch.Tensor
        Network output (velocity).
    target : torch.Tensor
        Ground-truth velocity (same shape as ``pred``).
    delta : float
        Transition point; ≈ median absolute residual of the L1 arm (≈ 0.90).

    Returns
    -------
    torch.Tensor
        Element-wise loss, same shape as ``pred``.
    """
    r = pred - target
    return delta**2 * (torch.sqrt(1.0 + (r / delta) ** 2) - 1.0)


class CFMLoss(AbstractFMLoss):
    r"""Rectified-flow regression loss on the velocity field.

    :math:`\mathcal{L}_\text{CFM} = \mathbb{E}\left[ \| G_\theta(x_t, t, c) -
    (x_1 - x_0) \|_p^p \right]`

    Parameters
    ----------
    reduction : {"none", "mean", "sum"}
        ``"mean"`` reduces over batch and spatial dims (S1 v2 default);
        ``"sum"`` sums (useful only for testing scale invariants);
        ``"none"`` is required when any region-weights mode is enabled — the
        per-voxel loss is multiplied by the region-weight tensor and reduced
        as ``(loss * w).sum() / w.sum()``.
    norm : {"l2", "l1", "huber"}
        ``"l1"`` matches T1C-RFlow and the S1 v2 baseline; ``"l2"`` is the
        constructor default (kept for backward compatibility with tests that
        do not set the field); ``"huber"`` is pseudo-Huber (Song & Dhariwal
        ICLR 2024), used for arm C of the §18 ablation.
    delta : float
        Transition point for the pseudo-Huber loss.  Pre-registered from the
        L1 arm's median absolute velocity residual (≈ 0.90).  Ignored when
        ``norm`` is ``"l2"`` or ``"l1"``.
    region_weights : RegionWeights | None
        Sub-region mode (S1 v3b_rw).  When non-None and ``enabled=True``,
        requires ``reduction="none"``.  Byte-identical to legacy path when
        None or disabled.
    brain_tc_weights : BrainTCWeights | None
        Brain/TC three-partition mode.  Mutually exclusive with
        ``region_weights`` being enabled.  When non-None and ``enabled=True``,
        requires ``reduction="none"`` and ``inputs.m_tc_soft`` non-None.

    Raises
    ------
    ValueError
        If ``reduction != "none"`` while any enabled region-weights config is
        supplied, or if both ``region_weights`` and ``brain_tc_weights`` are
        simultaneously enabled.
    """

    def __init__(
        self,
        reduction: str = "mean",
        norm: str = "l2",
        delta: float = 0.90,
        region_weights: RegionWeights | None = None,
        brain_tc_weights: BrainTCWeights | None = None,
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum'; got {reduction!r}")
        if norm not in ("l2", "l1", "huber"):
            raise ValueError(f"norm must be 'l2', 'l1', or 'huber'; got {norm!r}")
        rw_active = region_weights is not None and region_weights.enabled
        btc_active = brain_tc_weights is not None and brain_tc_weights.enabled
        if (rw_active or btc_active) and reduction != "none":
            raise ValueError(
                f"region_weights / brain_tc_weights enabled=True requires "
                f"reduction='none'; got reduction={reduction!r}"
            )
        if rw_active and btc_active:
            raise ValueError(
                "region_weights and brain_tc_weights cannot both be enabled "
                "simultaneously; use exactly one mode."
            )
        self.reduction = reduction
        self.norm = norm
        self.delta = delta
        self.region_weights = region_weights
        self.brain_tc_weights = brain_tc_weights

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        rw_active = self.region_weights is not None and self.region_weights.enabled
        btc_active = self.brain_tc_weights is not None and self.brain_tc_weights.enabled

        if rw_active or btc_active:
            # Compute per-voxel loss tensor — same for both region modes.
            if self.norm == "l2":
                voxel = F.mse_loss(inputs.v_orig, inputs.u_target, reduction="none")
            elif self.norm == "huber":
                voxel = _pseudo_huber(inputs.v_orig, inputs.u_target, self.delta)
            else:
                voxel = F.l1_loss(inputs.v_orig, inputs.u_target, reduction="none")

            if btc_active:
                # Brain/TC three-partition mode (default: brain=1, tc=1 ≡ mean L1).
                assert self.brain_tc_weights is not None  # guarded by btc_active
                w = build_brain_tc_weight_tensor(
                    self.brain_tc_weights,
                    inputs.m_brain,
                    inputs.m_tc_soft,
                    channels=voxel.shape[1],
                )
            else:
                # Sub-region mode (legacy S1 v3b_rw).
                assert self.region_weights is not None  # guarded by rw_active
                w = build_region_weight_tensor(
                    self.region_weights,
                    inputs.m_brain,
                    inputs.m_tumor,
                    channels=voxel.shape[1],
                )

            assert w is not None, "build_*_weight_tensor returned None despite enabled=True"
            # (loss * w).sum() / w.sum() — clamp the denominator only at
            # floor-numerical level so a degenerate all-zero-weight tensor
            # surfaces as a non-finite loss rather than a silent 0.
            return (voxel * w).sum() / w.sum().clamp_min(1e-12)

        # Legacy mean/sum path — no region weighting.
        if self.norm == "l2":
            return F.mse_loss(inputs.v_orig, inputs.u_target, reduction=self.reduction)
        if self.norm == "huber":
            loss = _pseudo_huber(inputs.v_orig, inputs.u_target, self.delta)
            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss  # "none"
        return F.l1_loss(inputs.v_orig, inputs.u_target, reduction=self.reduction)

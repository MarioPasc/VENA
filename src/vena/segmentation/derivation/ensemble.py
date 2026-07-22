"""K-fold ensemble aggregation of soft mask predictions.

The primary output is the channel-wise mean over K fold-models.
Optionally, per-voxel k-fold disagreement (std over the fold axis,
averaged across the two class channels to a single scalar per voxel) is
appended for ablation analysis.

**Label the extra channel "k-fold disagreement"** — NOT epistemic uncertainty
(B.f-§3).  It measures spread across the fold ensemble, not calibrated
epistemic uncertainty from a Bayesian posterior.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from vena.segmentation.exceptions import SegDerivationError


def ensemble_soft(
    maps: Sequence[Tensor],
    *,
    emit_variance: bool = False,
) -> Tensor:
    """Aggregate K fold-model soft masks by mean (and optionally k-fold std).

    Parameters
    ----------
    maps : Sequence[Tensor]
        K soft probability maps, each of shape ``(2, *spatial)``.
        All maps must have identical shape.  Channel 0 = WT, channel 1 = NETC.
    emit_variance : bool, optional
        If ``True``, append a k-fold disagreement channel (per-voxel std
        averaged over the two class channels) so that output shape is
        ``(3, *spatial)``.  If ``False`` (default), return only the mean
        with shape ``(2, *spatial)``.

    Returns
    -------
    Tensor
        * ``emit_variance=False``: mean over K fold-models, shape
          ``(2, *spatial)``.
        * ``emit_variance=True``: ``[mean_wt, mean_netc, kfold_disagreement]``
          concatenated along dim 0, shape ``(3, *spatial)``.
          The disagreement channel is the per-class std averaged over channels
          and is always ``≥ 0``; it is zero when all K fold-models agree.

    Raises
    ------
    SegDerivationError
        If ``maps`` is empty, any map does not have exactly 2 channels,
        or maps have inconsistent shapes.
    """
    if len(maps) == 0:
        raise SegDerivationError("ensemble_soft requires at least one map")

    reference_shape = maps[0].shape
    if reference_shape[0] != 2:
        raise SegDerivationError(
            f"each map must have exactly 2 channels (WT, NETC); got {reference_shape[0]}"
        )
    for i, m in enumerate(maps):
        if m.shape != reference_shape:
            raise SegDerivationError(
                f"map {i} shape {tuple(m.shape)} != reference shape {tuple(reference_shape)}"
            )

    stacked = torch.stack(list(maps), dim=0)  # (K, 2, *spatial)
    mean = stacked.mean(dim=0)  # (2, *spatial)

    if not emit_variance:
        return mean

    # Per-class std over the K fold-models → (2, *spatial).
    # For K=1 std is undefined; return zeros (no disagreement with a single model).
    if stacked.shape[0] == 1:
        std_map = torch.zeros_like(mean[:1])  # (1, *spatial)
    else:
        per_class_std = stacked.std(dim=0, unbiased=True)  # (2, *spatial)
        # Average the two class stds to a single "k-fold disagreement" scalar
        # per voxel.  This gives a channel-agnostic measure of fold spread.
        std_map = per_class_std.mean(dim=0, keepdim=True)  # (1, *spatial)

    return torch.cat([mean, std_map], dim=0)  # (3, *spatial)


__all__ = ["ensemble_soft"]

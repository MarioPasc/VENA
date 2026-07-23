"""Hard-mask overlap metrics for the tumour segmenter.

Public API: :func:`dice`, :func:`average_hausdorff`, :func:`et_diagnostic`.

**Threshold convention**: all overlap functions threshold the soft prediction at
0.5 before computing hard-mask metrics.  Calibration metrics
(``calibration.py``) operate on raw soft probabilities — never thresholded.

**Empty-mask contract**:

- Two empty masks  → Dice = 1.0 (both correctly predict nothing; MONAI's
  default of 0.0 violates this semantic).
- AHD undefined when either mask is empty → returns ``float("nan")``.
  Callers must exclude NaN from aggregation (``nanmean`` pattern).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from monai.metrics import compute_hausdorff_distance

from vena.segmentation.exceptions import SegMetricError

if TYPE_CHECKING:
    from torch import Tensor


def dice(
    pred_soft: Tensor,
    target_hard: Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    """Sørensen–Dice coefficient for a single class (hard-mask form).

    Parameters
    ----------
    pred_soft : Tensor
        Soft prediction, any shape, values in [0, 1].
    target_hard : Tensor
        Binary ground-truth, same shape as *pred_soft*, values in {0, 1}.
    threshold : float
        Applied to *pred_soft* to obtain the hard binary prediction.

    Returns
    -------
    float
        Dice in [0, 1].  Two empty masks return 1.0 (both correctly predict
        background — the standard convention for empty-class evaluation).

    Raises
    ------
    SegMetricError
        If *pred_soft* and *target_hard* have different shapes.
    """
    if pred_soft.shape != target_hard.shape:
        raise SegMetricError(
            f"Shape mismatch: pred {pred_soft.shape} vs target {target_hard.shape}"
        )

    pred = (pred_soft >= threshold).float()
    target = target_hard.float()

    intersection = (pred * target).sum()
    denom = pred.sum() + target.sum()

    if denom == 0:
        # Both masks are empty: perfect agreement on background.
        return 1.0

    return float(2.0 * intersection / denom)


def _ensure_5d(t: Tensor) -> Tensor:
    """Add leading dims until the tensor is 5-D (B, C, H, W, D) for MONAI."""
    while t.dim() < 5:
        t = t.unsqueeze(0)
    return t


def average_hausdorff(
    pred_soft: Tensor,
    target_hard: Tensor,
    *,
    threshold: float = 0.5,
    percentile: float = 95,
) -> float:
    """Percentile Hausdorff distance (HD95 by default) for a single class.

    Despite its name, this function computes the *percentile* Hausdorff
    distance via MONAI ``compute_hausdorff_distance(percentile=percentile)``,
    not the true Average Surface Distance (ASD).  The 95th-percentile variant
    (HD95) is the community standard in BraTS and most clinical segmentation
    benchmarks, and is more robust to outliers than the strict max-HD.

    Parameters
    ----------
    pred_soft : Tensor
        Soft prediction, any shape.  Thresholded at *threshold* internally.
    target_hard : Tensor
        Binary ground-truth, same shape as *pred_soft*.
    threshold : float
        Applied to *pred_soft* to produce the hard prediction.
    percentile : float
        Percentile for MONAI's Hausdorff implementation.  Default 95 → HD95.

    Returns
    -------
    float
        HD_{percentile} in voxels.  Returns ``float("nan")`` when either mask
        is empty — callers must exclude NaN from aggregation.

    Raises
    ------
    SegMetricError
        If *pred_soft* and *target_hard* have different shapes.
    """
    if pred_soft.shape != target_hard.shape:
        raise SegMetricError(
            f"Shape mismatch: pred {pred_soft.shape} vs target {target_hard.shape}"
        )

    pred = (pred_soft >= threshold).float()
    target = target_hard.float()

    # HD is undefined when either surface is empty.
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")

    # MONAI expects (B, C, spatial…) — ensure 5-D.
    p5 = _ensure_5d(pred)
    t5 = _ensure_5d(target)

    hd = compute_hausdorff_distance(
        p5,
        t5,
        include_background=True,
        percentile=percentile,
    )
    val = hd.item()
    return float("nan") if math.isnan(val) else float(val)


def et_diagnostic(
    pred: Tensor,
    target: Tensor,
) -> dict[str, float]:
    """ET (Enhancing Tumour) diagnostic — **reported, NOT part of the G-SEG gate**.

    Computes ET = ``clip(TC − NETC, 0, 1)`` from the predicted and target
    2-channel soft/hard maps, then returns the ET-Dice and the mean soft ET
    probability within the target ET region.  ET is the load-bearing enhancing
    ring that the generator must reproduce; NETC miscalibration that corrupts it
    must be visible (design authority B.c, iter-9 §a).

    Parameters
    ----------
    pred : Tensor
        Soft 2-channel prediction, shape ``(2, *spatial)``.
        Channel 0 = TC (tumour core), channel 1 = NETC.
    target : Tensor
        Hard 2-channel ground-truth, same shape as *pred*.
        Channel 0 = TC, channel 1 = NETC (binary, 0/1).

    Returns
    -------
    dict[str, float]
        ``"et_dice"`` : Dice of ``ET = clip(TC − NETC, 0, 1)`` thresholded at 0.5.
        ``"mean_et_soft"`` : Mean soft ET probability in target-ET voxels.
          ``float("nan")`` when the target ET region is empty.

    Raises
    ------
    SegMetricError
        If *pred* does not have exactly 2 channels in its first dimension.
    """
    if pred.shape[0] != 2 or target.shape[0] != 2:
        raise SegMetricError(
            f"Expected 2 channels in dim 0; got pred {pred.shape}, target {target.shape}"
        )
    if pred.shape != target.shape:
        raise SegMetricError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    # Soft ET prediction: clip(TC − NETC, 0, 1).
    pred_et = (pred[0] - pred[1]).clamp(0.0, 1.0)
    # Hard ET target derived from the hard TC/NETC channels.
    target_et = (target[0].float() - target[1].float()).clamp(0.0, 1.0)

    et_dice_val = dice(pred_et, target_et, threshold=0.5)

    # Mean soft ET probability inside the true ET region.
    et_mask = target_et > 0.5
    if et_mask.any():
        mean_et_soft = float(pred_et[et_mask].mean().item())
    else:
        mean_et_soft = float("nan")

    return {
        "et_dice": et_dice_val,
        "mean_et_soft": mean_et_soft,
    }

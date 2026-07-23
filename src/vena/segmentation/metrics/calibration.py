"""Calibration metrics for the tumour segmenter.

Public API: :func:`expected_calibration_error`, :func:`classwise_ece`,
:func:`brier`.

**Critical convention**: ALL calibration metrics operate on **raw soft
probabilities** — never on thresholded binary masks.  Thresholding destroys the
probability information that calibration is designed to measure.  The overlap
module (``overlap.py``) handles hard-mask metrics.

The generator consumes soft probability maps directly, so calibration is
load-bearing: a well-calibrated segmenter passes more informative priors to the
conditioning branch than an overconfident one (design authority B.f-§7).

Temperature scaling is **not applied here**.  Calibration is *measured* in this
module; the correction decision (dropped in planning Q5) is not implemented.
"""

from __future__ import annotations

import torch

from vena.segmentation.exceptions import SegMetricError


def _ece_from_flat(
    probs_flat: torch.Tensor,
    targets_flat: torch.Tensor,
    n_bins: int,
) -> float:
    """Inner ECE computation on flattened 1-D tensors.

    Parameters
    ----------
    probs_flat : Tensor
        Raw predicted probabilities, shape ``(N,)``, values in [0, 1].
    targets_flat : Tensor
        Binary ground-truth, shape ``(N,)``.
    n_bins : int
        Number of equal-width probability bins covering [0, 1].

    Returns
    -------
    float
        ECE in [0, 1].
    """
    n = probs_flat.numel()
    if n == 0:
        raise SegMetricError("Empty input tensors for ECE computation")

    ece = 0.0
    bin_width = 1.0 / n_bins

    for i in range(n_bins):
        lo = i * bin_width
        hi = (i + 1) * bin_width
        # The last bin is closed on the right to capture prob == 1.0.
        if i < n_bins - 1:
            in_bin = (probs_flat >= lo) & (probs_flat < hi)
        else:
            in_bin = (probs_flat >= lo) & (probs_flat <= hi)

        n_bin = int(in_bin.sum().item())
        if n_bin == 0:
            continue

        conf = float(probs_flat[in_bin].mean().item())
        acc = float(targets_flat[in_bin].mean().item())
        weight = n_bin / n
        ece += weight * abs(conf - acc)

    return ece


def expected_calibration_error(
    probs: torch.Tensor,
    target_hard: torch.Tensor,
    *,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) on raw soft probabilities.

    Flattens all channels and spatial dimensions together, then computes ECE
    over the pooled population:

    .. math::
        \\text{ECE} = \\sum_{b=1}^{B} \\frac{|\\mathcal{B}_b|}{N}
                      \\left|\\operatorname{acc}(\\mathcal{B}_b)
                             - \\operatorname{conf}(\\mathcal{B}_b)\\right|

    For per-class ECE use :func:`classwise_ece`.

    Parameters
    ----------
    probs : Tensor
        Raw soft probabilities, any shape, values in [0, 1].
        **Never thresholded** — that is the point of measuring calibration.
    target_hard : Tensor
        Binary ground-truth, same shape as *probs*.
    n_bins : int
        Number of equal-width bins in [0, 1].

    Returns
    -------
    float
        ECE in [0, 1].  Smaller is better (0 = perfect calibration).

    Raises
    ------
    SegMetricError
        If *probs* and *target_hard* have different shapes or are empty.
    """
    if probs.shape != target_hard.shape:
        raise SegMetricError(f"Shape mismatch: probs {probs.shape} vs target {target_hard.shape}")

    return _ece_from_flat(
        probs.float().flatten(),
        target_hard.float().flatten(),
        n_bins,
    )


def classwise_ece(
    probs: torch.Tensor,
    target_hard: torch.Tensor,
    *,
    n_bins: int = 15,
) -> dict[str, float]:
    """Per-class Expected Calibration Error.

    Computes ECE independently for the TC and NETC channels.

    Parameters
    ----------
    probs : Tensor
        Raw soft probabilities, shape ``(2, *spatial)``.
        Channel 0 = TC, channel 1 = NETC.
    target_hard : Tensor
        Binary ground-truth, same shape as *probs*.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    dict[str, float]
        ``{"tc": ECE_TC, "netc": ECE_NETC}``.

    Raises
    ------
    SegMetricError
        If *probs* does not have exactly 2 channels or shapes mismatch.
    """
    if probs.shape != target_hard.shape:
        raise SegMetricError(f"Shape mismatch: probs {probs.shape} vs target {target_hard.shape}")
    if probs.shape[0] != 2:
        raise SegMetricError(
            f"Expected 2 channels in dim 0 for classwise ECE; got {probs.shape[0]}"
        )

    p = probs.float()
    t = target_hard.float()
    return {
        "tc": _ece_from_flat(p[0].flatten(), t[0].flatten(), n_bins),
        "netc": _ece_from_flat(p[1].flatten(), t[1].flatten(), n_bins),
    }


def brier(
    probs: torch.Tensor,
    target_hard: torch.Tensor,
) -> dict[str, float]:
    """Per-class Brier score (mean squared error of probabilities).

    .. math::
        \\text{BS}_c = \\frac{1}{N} \\sum_{i=1}^{N} (p_{c,i} - y_{c,i})^2

    Parameters
    ----------
    probs : Tensor
        Raw soft probabilities, shape ``(2, *spatial)``.
        Channel 0 = TC, channel 1 = NETC.  **Never thresholded.**
    target_hard : Tensor
        Binary ground-truth, same shape as *probs*.

    Returns
    -------
    dict[str, float]
        ``{"tc": BS_TC, "netc": BS_NETC}``.  Lower is better (0 = perfect).

    Raises
    ------
    SegMetricError
        If *probs* does not have exactly 2 channels or shapes mismatch.
    """
    if probs.shape != target_hard.shape:
        raise SegMetricError(f"Shape mismatch: probs {probs.shape} vs target {target_hard.shape}")
    if probs.shape[0] != 2:
        raise SegMetricError(f"Expected 2 channels in dim 0 for Brier score; got {probs.shape[0]}")

    p = probs.float()
    t = target_hard.float()

    bs_tc = float(((p[0] - t[0]) ** 2).mean().item())
    bs_netc = float(((p[1] - t[1]) ** 2).mean().item())

    return {"tc": bs_tc, "netc": bs_netc}

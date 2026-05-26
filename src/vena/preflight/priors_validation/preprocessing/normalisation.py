"""Robust z-score normalisation used by Test T3."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def robust_zscore(
    volume: NDArray[np.floating],
    mask: NDArray[np.bool_] | NDArray[np.integer],
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Robust z-score within ``mask`` using median and MAD.

    Returns a float32 array of the same shape as ``volume``; voxels outside
    the mask are set to zero. MAD is scaled by 1.4826 so the result is
    Normal-equivalent (Rousseeuw & Croux 1993).
    """
    arr = np.asarray(volume, dtype=np.float32)
    mask_bool = np.asarray(mask) > 0
    if not mask_bool.any():
        return np.zeros_like(arr, dtype=np.float32)
    in_mask = arr[mask_bool]
    med = float(np.median(in_mask))
    mad = float(np.median(np.abs(in_mask - med)))
    scale = 1.4826 * mad if mad > 0 else (float(in_mask.std()) or eps)
    out = (arr - med) / max(scale, eps)
    out = out.astype(np.float32)
    out *= mask_bool.astype(np.float32)
    return out

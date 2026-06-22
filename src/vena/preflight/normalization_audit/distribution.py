"""Distribution-shape audit: per-modality histograms + KL divergence vs V0.

The MAISI-V2 VAE was pre-trained on a specific normalised-intensity
distribution. Pushing a variant whose normalised distribution is very
different from V0 risks moving the encoder out of distribution, which
shows up as elevated reconstruction error. ``KL(V_i || V0)`` per modality
is the early-warning signal.
"""

from __future__ import annotations

import math

import numpy as np
import torch

HIST_BINS: int = 256
HIST_RANGE: tuple[float, float] = (-0.1, 1.5)
HIST_EPS: float = 1e-10


def histogram_normalised(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    bins: int = HIST_BINS,
    range_: tuple[float, float] = HIST_RANGE,
) -> np.ndarray:
    """Compute a probability histogram of foreground voxel intensities.

    Parameters
    ----------
    x : torch.Tensor
        Volume of shape ``(1, 1, H, W, D)`` (other shapes are flattened).
    mask : torch.Tensor | None
        Optional boolean mask selecting which voxels to histogram.
        When ``None`` the heuristic ``x > 0`` is used.
    """
    flat = x.flatten()
    if mask is None:
        sel = flat[flat > 0]
    else:
        m = mask.flatten() > 0
        sel = flat[m]
    if sel.numel() == 0:
        return np.zeros(bins, dtype=np.float64) + HIST_EPS / bins
    hist = torch.histc(sel, bins=bins, min=range_[0], max=range_[1])
    p = hist.detach().cpu().double().numpy()
    p = p + HIST_EPS  # avoid log(0)
    p = p / p.sum()
    return p


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Discrete KL divergence ``KL(p || q)`` in nats.

    Both inputs are probability distributions; ``HIST_EPS`` smoothing
    guarantees positivity.
    """
    if p.shape != q.shape:
        raise ValueError(f"kl_divergence: shape mismatch — p {p.shape} vs q {q.shape}")
    if not math.isclose(p.sum(), 1.0, abs_tol=1e-6):
        p = p / p.sum()
    if not math.isclose(q.sum(), 1.0, abs_tol=1e-6):
        q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


__all__ = [
    "HIST_BINS",
    "HIST_RANGE",
    "histogram_normalised",
    "kl_divergence",
]

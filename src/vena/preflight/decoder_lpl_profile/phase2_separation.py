"""Phase 2 — pre/post separation, error concentration, t-sweep (§4.2 + §4.4).

Per (cohort, patient, variant) cell, with the S1 model already loaded
externally:

1. Decode ``z_t1c`` and ``z_t1pre`` through the feature extractor at
   blocks ``{0..5}`` and compute the per-region feature distance —
   this is the §4.2 pre/post separation.
2. Sample ``x̂_1^{S1}`` at NFE=10 Euler via the S1 model's ``ema_call``;
   decode and compute the per-region residual ``||phi(z_t1c) - phi(x̂_1^{S1})||``
   — the §4.2 error concentration.
3. For each ``t`` in the configured sweep, build ``x_t = (1-t) * x_0 +
   t * z_t1c``, compute ``x̂_1(t) = x_t + (1-t) * G_theta(x_t, t, c)``
   (one trunk forward per t) and the per-block feature distance to
   ``z_t1c`` — the §4.4 ``x̂_1`` reliability curve. The knee picks
   ``t_min``.

This phase needs the S1 sampler + a velocity-predictor closure; the
engine constructs both and hands them in. ``model_call(x_t, t,
conditioning)`` returns the velocity; ``sampler(model_call, x_0, nfe)``
returns the final sample.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

FeatureExtractor = Callable[[torch.Tensor], dict[int, torch.Tensor]]
ModelCall = Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]
Sampler = Callable[[ModelCall, torch.Tensor, int], torch.Tensor]


def _resample_mask_to(feat_shape: tuple[int, int, int], mask: torch.Tensor) -> torch.Tensor:
    """NN-upsample a latent-res mask to the block's spatial grid."""
    if mask.shape[-3:] == feat_shape:
        return mask
    return F.interpolate(mask.float(), size=feat_shape, mode="nearest")


def _per_region_feat_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    m_wt: torch.Tensor,
    m_brain: torch.Tensor,
) -> dict[str, float]:
    """``|a - b|`` aggregated over (WT, notWT, global) per the §4.2 convention.

    The mean is taken over the channel-axis-L2-norm of the per-voxel
    difference, restricted by the region indicator.
    """
    diff = a - b
    voxel_norm = diff.float().pow(2).sum(dim=1, keepdim=True).sqrt()  # (B, 1, ...)
    feat_shape = a.shape[-3:]
    mwt = _resample_mask_to(feat_shape, m_wt)
    mbrain = _resample_mask_to(feat_shape, m_brain)
    is_wt = (mwt >= 0.5).float() * mbrain
    not_wt = (1.0 - (mwt >= 0.5).float()) * mbrain
    eps = 1e-8

    def _masked_mean(weights: torch.Tensor) -> float:
        num = (voxel_norm * weights).sum()
        den = weights.sum().clamp(min=eps)
        return float((num / den).item())

    return {
        "WT": _masked_mean(is_wt),
        "notWT": _masked_mean(not_wt),
        "global": _masked_mean(mbrain.float()),
    }


def pre_post_separation(
    extract: FeatureExtractor,
    z_t1c: torch.Tensor,
    z_t1pre: torch.Tensor,
    *,
    blocks: tuple[int, ...],
    m_wt_lat: torch.Tensor,
    m_brain_lat: torch.Tensor,
) -> dict[int, dict[str, float]]:
    """Per-block per-region feature distance ``||phi(z_t1c) - phi(z_t1pre)||``.

    Drives the §4.2 ``A`` selection — blocks whose pre/post separation
    is largest *relative to within-class variance* carry enhancement
    signal.
    """
    with torch.no_grad():
        feats_post = extract(z_t1c)
        feats_pre = extract(z_t1pre)
    out: dict[int, dict[str, float]] = {}
    for blk in blocks:
        out[blk] = _per_region_feat_distance(feats_post[blk], feats_pre[blk], m_wt_lat, m_brain_lat)
    return out


def error_concentration(
    extract: FeatureExtractor,
    z_t1c: torch.Tensor,
    x1_hat_s1: torch.Tensor,
    *,
    blocks: tuple[int, ...],
    m_wt_lat: torch.Tensor,
    m_brain_lat: torch.Tensor,
) -> dict[int, dict[str, float]]:
    """Per-block per-region residual ``||phi(z_t1c) - phi(x̂_1^{S1})||``.

    Locates where the S1 model's prediction is wrong by depth × region —
    the blocks S3 should weight most.
    """
    with torch.no_grad():
        feats_target = extract(z_t1c)
        feats_pred = extract(x1_hat_s1)
    out: dict[int, dict[str, float]] = {}
    for blk in blocks:
        out[blk] = _per_region_feat_distance(
            feats_target[blk], feats_pred[blk], m_wt_lat, m_brain_lat
        )
    return out


def x1_reliability_vs_t(
    extract: FeatureExtractor,
    model_call: ModelCall,
    z_t1c: torch.Tensor,
    conditioning: dict,
    *,
    t_sweep: tuple[float, ...],
    blocks: tuple[int, ...],
    m_brain_lat: torch.Tensor,
    rng: torch.Generator | None = None,
) -> dict[float, dict[int, float]]:
    """§4.4 ``x̂_1`` reliability — per-block per-t global-brain residual.

    For each ``t``: sample ``x_0 ~ N(0, I)``, build the interpolant
    ``x_t = (1-t) x_0 + t z_t1c``, compute the velocity ``G(x_t, t, c)``,
    then ``x̂_1(t) = x_t + (1-t) * G``. The reliability metric is
    ``||phi(z_t1c) - phi(x̂_1(t))||`` per block, averaged over brain
    foreground only (the per-region split is not needed here — the knee
    is a global-volume property).

    The function returns nested ``{t: {block: distance}}`` so the
    aggregator can fit a knee-detection over the per-block curves.
    """
    if rng is None:
        rng = torch.Generator(device=z_t1c.device)
        rng.manual_seed(0)
    out: dict[float, dict[int, float]] = {}
    for t_val in t_sweep:
        t_tensor = torch.full(
            (z_t1c.shape[0],), float(t_val), device=z_t1c.device, dtype=z_t1c.dtype
        )
        x0 = torch.randn(z_t1c.shape, generator=rng, device=z_t1c.device, dtype=z_t1c.dtype)
        # Broadcast scalar t over batch dim for the interpolant.
        t_b = t_tensor.view(-1, *([1] * (z_t1c.ndim - 1)))
        x_t = (1.0 - t_b) * x0 + t_b * z_t1c
        with torch.no_grad():
            v = model_call(x_t, t_tensor, conditioning)
            x1_hat = x_t + (1.0 - t_b) * v
            feats_target = extract(z_t1c)
            feats_pred = extract(x1_hat)
        per_block: dict[int, float] = {}
        for blk in blocks:
            diff = feats_target[blk] - feats_pred[blk]
            voxel_norm = diff.float().pow(2).sum(dim=1, keepdim=True).sqrt()
            mbrain = _resample_mask_to(feats_target[blk].shape[-3:], m_brain_lat)
            num = (voxel_norm * mbrain.float()).sum()
            den = mbrain.float().sum().clamp(min=1e-8)
            per_block[blk] = float((num / den).item())
        out[float(t_val)] = per_block
    return out


def detect_t_min_knee(reliability: dict[float, dict[int, float]]) -> float:
    """Pick ``t_min`` at the knee of the per-block reliability curve.

    Strategy: collapse per-t to a scalar (mean over blocks of the
    reliability metric), then take the ``t`` where the *second* finite
    difference is most negative — the curvature maximum on the
    monotonically-decreasing curve. Falls back to Berrada's default
    ``0.7`` on degenerate input.
    """
    ts = sorted(reliability)
    if len(ts) < 3:
        return 0.7
    means = np.array([float(np.mean(list(reliability[t].values()))) for t in ts])
    # Smooth via a 3-pt moving average to dampen sampler noise.
    if means.size >= 3:
        means = np.convolve(means, np.ones(3) / 3.0, mode="same")
    # Curvature = second discrete derivative.
    if means.size < 3:
        return 0.7
    curvature = np.diff(means, n=2)
    knee_idx = int(np.argmin(curvature)) + 1  # +1 because diff(n=2) shrinks index
    return float(ts[knee_idx])


__all__ = [
    "FeatureExtractor",
    "ModelCall",
    "Sampler",
    "detect_t_min_knee",
    "error_concentration",
    "pre_post_separation",
    "x1_reliability_vs_t",
]

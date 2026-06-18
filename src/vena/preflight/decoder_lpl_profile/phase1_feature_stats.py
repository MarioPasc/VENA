"""Phase 1 — per-block feature-statistics sweep (§4.1).

Per (cohort, patient, variant) cell:

1. Read the target latent ``z_t1c`` (clean v0 from the cohort H5; v1..v4
   from the augmented H5 keyed by ``variants``).
2. Decode through the feature extractor at blocks ``{0..5}``.
3. Accumulate three CSV tables:
   * ``per_block_magnitude.csv`` — mean / std / p99 of ``||phi_l||`` per
     block per patient, across spatial × channel axes.
   * ``per_channel_L_dec_distribution.csv`` — per-block per-channel
     mean / p99 / MAD of ``||phi_l||`` (the channel-concentration curve).
   * ``outlier_threshold.csv`` — per-block per-channel MAD with the
     recommended ``k`` (Berrada default 5).

Forward-only — no backward graph, no S1 sampling. Cheapest phase to run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

# Type alias for the partial-decode closure handed in by the engine.
FeatureExtractor = Callable[[torch.Tensor], dict[int, torch.Tensor]]


def _flatten_per_channel(feat: torch.Tensor) -> torch.Tensor:
    """``(B, C, ...)`` → ``(B*S, C)`` where ``S`` is the spatial product."""
    return feat.movedim(1, -1).reshape(-1, feat.shape[1])


def _mad(values: torch.Tensor, dim: int) -> torch.Tensor:
    med = values.median(dim=dim, keepdim=True).values
    return (values - med).abs().median(dim=dim, keepdim=True).values


def per_patient_block_magnitude(
    extract: FeatureExtractor,
    z_target: torch.Tensor,
    *,
    blocks: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    """Decode ``z_target`` through the extractor and return per-block stats.

    Returns
    -------
    dict[block_idx, {mean_norm, std_norm, p99_norm}]
        Stats computed on the channel-flat L2-norm distribution:
        ``norm_voxel = ||phi[B, :, x, y, z]||_2 over channels``.
    """
    with torch.no_grad():
        feats = extract(z_target)
    out: dict[int, dict[str, float]] = {}
    for blk in blocks:
        feat = feats[blk].float()
        # Channel-collapse to per-voxel L2 norm so the magnitude is a
        # single scalar per voxel (same convention §4.1 reports against).
        norms = feat.pow(2).sum(dim=1).sqrt().flatten()  # (B * H * W * D,)
        out[blk] = {
            "mean_norm": float(norms.mean().item()),
            "std_norm": float(norms.std(unbiased=False).item()),
            "p99_norm": float(torch.quantile(norms, 0.99).item()),
        }
    return out


def per_channel_feature_stats(
    extract: FeatureExtractor,
    z_target: torch.Tensor,
    *,
    blocks: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    """Per-block per-channel feature distribution.

    For each block we compute, across the spatial dimensions of the batch,
    three per-channel scalars:

    * ``mean_abs`` — ``E[|phi_{l,c}|]`` per channel.
    * ``p99_abs`` — 99th percentile of ``|phi_{l,c}|`` per channel.
    * ``mad`` — median absolute deviation per channel (drives the
      per-block ``outlier_k`` decision in §4.1 / Berrada §3.4).

    Returns
    -------
    dict[block_idx, {mean_abs, p99_abs, mad}]
        Each value is a NumPy array of shape ``(C_block,)``.
    """
    with torch.no_grad():
        feats = extract(z_target)
    out: dict[int, dict[str, np.ndarray]] = {}
    for blk in blocks:
        feat = feats[blk].float()
        flat = _flatten_per_channel(feat).abs()  # (N, C)
        mean_abs = flat.mean(dim=0).cpu().numpy()
        p99_abs = torch.quantile(flat, 0.99, dim=0).cpu().numpy()
        # MAD on the signed standardised distribution is what Berrada §3.4
        # operates on; here we use MAD of |x| which is a strictly tighter
        # statistic and matches the outlier-mask threshold in
        # :func:`vena.model.fm.lpl.loss._outlier_mask`.
        mad_vals = _mad(flat, dim=0).abs().squeeze(0).cpu().numpy()
        out[blk] = {"mean_abs": mean_abs, "p99_abs": p99_abs, "mad": mad_vals}
    return out


def recommend_outlier_k(
    per_channel: dict[int, dict[str, np.ndarray]],
    *,
    default_k: float = 5.0,
    heavy_tail_threshold: float = 10.0,
) -> dict[int, float]:
    """Pick ``k`` per block from the per-channel ``p99 / MAD`` ratio.

    Heuristic: when the typical channel's ``p99 / (MAD + eps)`` ratio
    exceeds ``heavy_tail_threshold``, the distribution is heavy-tailed
    and a tighter ``k`` would mask out signal voxels alongside outliers.
    Default ``k = 5`` (Berrada's recommendation) is the floor.
    """
    out: dict[int, float] = {}
    eps = 1e-8
    for blk, stats in per_channel.items():
        ratio = stats["p99_abs"] / (stats["mad"] + eps)
        median_ratio = float(np.median(ratio))
        if median_ratio > heavy_tail_threshold:
            # Heavy-tailed → widen ``k`` proportionally so the threshold
            # tracks the actual tail. Capped at 10 to keep gradients tame.
            k = min(default_k * (median_ratio / heavy_tail_threshold), 10.0)
        else:
            k = default_k
        out[blk] = float(k)
    return out


__all__ = [
    "FeatureExtractor",
    "per_channel_feature_stats",
    "per_patient_block_magnitude",
    "recommend_outlier_k",
]

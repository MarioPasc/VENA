"""Region-mask resampling and weight-map helpers (§2.6 + §4.7a).

The §2.6 region-weighted LPL variant reads masks at *the spatial grid of
the decoder block* — i.e. block 2 at native latent resolution and block 5
at 2× latent resolution. The latent dataloader stores ``masks/tumor_latent``
(soft, 3-channel) and ``masks/brain_latent`` (hard, 1-channel) at native
latent res only (``LATENT_SPATIAL = (48, 56, 48)``). This module owns the
conventions for getting them to per-block grids without lossy stair-stepping.

The §4.7a-pinned conventions are:

* binary WT (the §2.6 default) → nearest-neighbour upsample, soft union
  thresholded at the §4.7a default 0.5;
* soft WT (the sweep variant) → trilinear upsample on the
  ``clip(sum(tumor_lat, axis=channels), 0, 1)``, multiplied by the
  brain mask;
* the brain mask is *always* nearest-neighbour (it is binary by
  construction).

The region weight map combines an alpha-weighted WT / not-WT split with
the brain mask so out-of-brain voxels contribute zero gradient.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def soft_wt_from_tumor_latent(
    tumor_lat: torch.Tensor,
    brain_lat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the §4.7a soft WT membership from the 3-channel ``tumor_latent``.

    Parameters
    ----------
    tumor_lat : torch.Tensor
        Shape ``(B, 3, H, W, D)`` — NETC / ED / ET soft probabilities at
        latent resolution.
    brain_lat : torch.Tensor | None
        Shape ``(B, 1, H, W, D)`` — hard binary brain mask. When supplied,
        the soft union is multiplied by ``brain_lat`` so out-of-brain
        voxels are zeroed.

    Returns
    -------
    torch.Tensor
        Shape ``(B, 1, H, W, D)`` — soft WT membership in ``[0, 1]``.
    """
    if tumor_lat.ndim != 5 or tumor_lat.shape[1] != 3:
        raise ValueError(f"tumor_lat must be (B, 3, H, W, D); got {tuple(tumor_lat.shape)}")
    soft = tumor_lat.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    if brain_lat is not None:
        if brain_lat.shape[0] != soft.shape[0] or brain_lat.shape[2:] != soft.shape[2:]:
            raise ValueError(
                f"brain_lat shape {tuple(brain_lat.shape)} incompatible with"
                f" tumor_lat-derived soft shape {tuple(soft.shape)}"
            )
        soft = soft * brain_lat
    return soft


def resample_region_to_block(
    mask: torch.Tensor,
    target_shape: tuple[int, int, int],
    *,
    mode: Literal["nearest", "trilinear"] = "nearest",
) -> torch.Tensor:
    """Resample a latent-resolution mask to a decoder block's spatial grid.

    Parameters
    ----------
    mask : torch.Tensor
        Shape ``(B, 1, H_lat, W_lat, D_lat)``.
    target_shape : (h, w, d)
        Target spatial shape of the block's activations.
    mode : Literal["nearest", "trilinear"], default "nearest"
        ``"nearest"`` for binary masks (preserves exact NN-upsample
        contract — a 1-voxel WT at latent corner upsamples to the right
        2×2×2 corner of block 5). ``"trilinear"`` for soft probability
        maps (smooths the boundary appropriately).

    Returns
    -------
    torch.Tensor
        Shape ``(B, 1, *target_shape)`` on the same device + dtype as
        ``mask``.
    """
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (B, 1, H, W, D); got {tuple(mask.shape)}")
    if mask.shape[2:] == target_shape:
        return mask
    if mode == "trilinear":
        # ``align_corners=False`` matches torch's default for spatial
        # operations on 5D and produces a uniform boundary smoothing.
        return F.interpolate(
            mask.float(), size=target_shape, mode="trilinear", align_corners=False
        ).to(mask.dtype)
    return F.interpolate(mask, size=target_shape, mode="nearest")


def region_weight_map(
    m_wt: torch.Tensor,
    m_brain: torch.Tensor,
    *,
    alpha_wt: float,
    alpha_notwt: float,
    soft: bool = False,
) -> torch.Tensor:
    """Per-voxel weight map for the region-weighted LPL.

    Binary mode (``soft=False``):
        ``w(x) = alpha_wt * 1_{m_wt(x) >= 0.5 ∧ m_brain(x) = 1}``
              ``+ alpha_notwt * 1_{m_wt(x) <  0.5 ∧ m_brain(x) = 1}``

    Soft mode (``soft=True``):
        ``w(x) = m_brain(x) * (alpha_wt * m_wt(x) + alpha_notwt * (1 - m_wt(x)))``

    Either way, out-of-brain voxels are zeroed.

    Parameters
    ----------
    m_wt : torch.Tensor
        Shape ``(B, 1, *block_shape)``. Hard binary in ``{0, 1}`` or soft
        in ``[0, 1]``.
    m_brain : torch.Tensor
        Shape ``(B, 1, *block_shape)``. Hard binary in ``{0, 1}``.
    alpha_wt : float
        WT region weight (per-region budget share before normalisation).
    alpha_notwt : float
        not-WT-in-brain region weight.
    soft : bool, default False
        Whether ``m_wt`` is to be treated as a soft probability.

    Returns
    -------
    torch.Tensor
        Shape ``(B, 1, *block_shape)``.
    """
    if m_wt.shape != m_brain.shape:
        raise ValueError(f"m_wt {tuple(m_wt.shape)} and m_brain {tuple(m_brain.shape)} must match")
    if soft:
        m_wt_use = m_wt.float().clamp(0.0, 1.0)
        weight = alpha_wt * m_wt_use + alpha_notwt * (1.0 - m_wt_use)
    else:
        is_wt = (m_wt >= 0.5).float()
        weight = alpha_wt * is_wt + alpha_notwt * (1.0 - is_wt)
    return weight * m_brain.float()

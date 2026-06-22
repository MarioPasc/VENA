"""Joint-modality percentile normalisation (variant V4).

The per-modality variants (V0, V1, V2, V3, V7, V8) all map each modality
independently to ``[0, 1]`` (or to a soft super-1 range with ``clip=False``).
This destroys the inter-modality intensity scale: a voxel that is bright
in T1c-post relative to T1pre at the same anatomical location ends up
near 1.0 in both modalities. The model can no longer "see" the contrast
that defines gadolinium enhancement.

Joint-modality normalisation computes a *single* ``(lo, hi)`` per patient
over the union of foreground voxels across all modalities, then applies
the same affine to each modality. T1c enhancement remains brighter than
T1pre at the same voxel in the normalised space, at the cost of T2/FLAIR
ending up in a narrower range (their dynamic range is naturally smaller).

This is the only variant that mechanically can clear C4 ≥ 1.5 (see
``.claude/notes/changes/2026-06-22_s1_v3_normalization_exploration.md``).
"""

from __future__ import annotations

import torch

from vena.model.autoencoder.maisi.exceptions import ShapeContractError


def joint_modality_percentile_normalise(
    images: dict[str, torch.Tensor],
    *,
    lower: float = 0.0,
    upper: float = 99.5,
    b_min: float = 0.0,
    b_max: float = 1.0,
    eps: float = 1e-8,
    clip: bool = True,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Map each modality into ``[b_min, b_max]`` using *one* joint (lo, hi).

    Parameters
    ----------
    images : dict[str, torch.Tensor]
        Modality → volume mapping. Every volume must share the same shape
        ``(B, 1, H, W, D)`` and be on the same device. Order of keys is
        preserved in the returned dict.
    lower, upper : float
        Percentile bounds in ``[0, 100]``, evaluated over the union of
        foreground voxels (per patient, across all modalities).
    b_min, b_max : float
        Output range.
    eps : float
        Numerical guard for empty / constant volumes.
    clip : bool
        Whether to clamp the rescaled values into ``[0, 1]`` before
        affine-mapping to ``[b_min, b_max]``. ``clip=False`` lets the
        bright tail keep its magnitude.
    mask : torch.Tensor | None
        Optional brain mask of shape ``(B, 1, H, W, D)``. When provided,
        only voxels with ``mask > 0`` contribute to the quantile estimate
        (across all modalities). When ``None``, the union of
        ``x_m > 0`` per modality is used as the foreground set.

    Returns
    -------
    dict[str, torch.Tensor]
        Modality → normalised volume, same keys / shapes / dtypes.

    Raises
    ------
    ShapeContractError
        If ``images`` is empty or modality shapes differ.
    """
    if not images:
        raise ShapeContractError("joint_modality_percentile_normalise: empty images dict")
    keys = list(images.keys())
    ref = images[keys[0]]
    if ref.ndim != 5:
        raise ShapeContractError(
            f"joint_modality_percentile_normalise expects (B,1,H,W,D); "
            f"got {tuple(ref.shape)} for modality '{keys[0]}'"
        )
    for k in keys[1:]:
        if tuple(images[k].shape) != tuple(ref.shape):
            raise ShapeContractError(
                f"joint_modality_percentile_normalise: shape mismatch — "
                f"'{keys[0]}'={tuple(ref.shape)} vs '{k}'={tuple(images[k].shape)}"
            )

    B = ref.shape[0]
    device = ref.device
    dtype = ref.dtype

    if mask is not None:
        if mask.ndim != 5 or mask.shape[0] != B or tuple(mask.shape[2:]) != tuple(ref.shape[2:]):
            raise ShapeContractError(
                f"joint_modality_percentile_normalise: mask shape {tuple(mask.shape)} "
                f"incompatible with image shape {tuple(ref.shape)}"
            )

    lo = torch.empty((B,), dtype=dtype, device=device)
    hi = torch.empty((B,), dtype=dtype, device=device)
    q = torch.tensor([lower / 100.0, upper / 100.0], dtype=dtype, device=device)

    for b in range(B):
        fg_voxels: list[torch.Tensor] = []
        for k in keys:
            vol = images[k][b, 0]
            if mask is not None:
                m = mask[b, 0] > 0
            else:
                m = vol > 0
            sel = vol[m]
            if sel.numel() > 0:
                fg_voxels.append(sel)
        if not fg_voxels:
            lo[b] = 0.0
            hi[b] = 1.0
            continue
        all_fg = torch.cat(fg_voxels)
        lh = torch.quantile(all_fg, q)
        lo[b] = lh[0]
        hi[b] = lh[1]

    out: dict[str, torch.Tensor] = {}
    for k in keys:
        x = images[k]
        denom = (hi - lo).clamp_min(eps).view(B, 1, 1, 1, 1)
        lo_b = lo.view(B, 1, 1, 1, 1)
        y = (x - lo_b) / denom
        if clip:
            y = y.clamp(0.0, 1.0)
        out[k] = y * (b_max - b_min) + b_min
    return out

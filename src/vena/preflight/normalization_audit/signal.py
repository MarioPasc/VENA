"""Image-space (C4) and latent-space (C5) signal preservation metrics.

Per-region ⟨|T1c − T1pre|⟩ on normalised image volumes, and
⟨|z_t1c − z_t1pre|⟩ on the encoded latents. The "C4/C5 winning condition"
is that the ET-vs-BNWT ratio is well above 1 — the encoder must "see"
more enhancement signal inside the enhancing region than in ordinary
non-tumour brain.

The diagnostic also returns ``⟨|z_t1c|⟩_ET`` and ``⟨|z_t1pre|⟩_ET``
separately so we can disambiguate whether the latent ratio improvement
comes from T1c rising, T1pre falling, or both (design-review addition).
"""

from __future__ import annotations

import torch

from .metrics import RegionMasks


def _broadcast_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    m = mask
    while m.ndim > like.ndim:
        m = m.squeeze(0)
    return m


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return float("nan")
    m = _broadcast_mask(mask, x)
    return float(x[m].mean().item())


def _masked_mean_abs(x: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return float("nan")
    m = _broadcast_mask(mask, x)
    return float(x[m].abs().mean().item())


def image_space_contrast(
    t1c_normalised: torch.Tensor,
    t1pre_normalised: torch.Tensor,
    regions: RegionMasks,
) -> dict[str, float]:
    """Per-region ⟨T1c⟩, ⟨T1pre⟩, ⟨|T1c − T1pre|⟩ in normalised image space.

    Returns one entry per region (et, netc, ed, bnwt, wt) for each of
    {``mean_t1c``, ``mean_t1pre``, ``mean_abs_diff``}.
    """
    diff = t1c_normalised - t1pre_normalised
    abs_diff = diff.abs()
    out: dict[str, float] = {}
    for name, mask in [
        ("et", regions.et),
        ("netc", regions.netc),
        ("ed", regions.ed),
        ("bnwt", regions.bnwt),
        ("wt", regions.wt),
    ]:
        out[f"image_mean_t1c_{name}"] = _masked_mean(t1c_normalised, mask)
        out[f"image_mean_t1pre_{name}"] = _masked_mean(t1pre_normalised, mask)
        out[f"image_mean_abs_diff_{name}"] = _masked_mean(abs_diff, mask)
    return out


def latent_space_contrast(
    z_t1c: torch.Tensor,
    z_t1pre: torch.Tensor,
    regions_latent: RegionMasks,
) -> dict[str, float]:
    """Per-region ⟨|z_t1c|⟩, ⟨|z_t1pre|⟩, ⟨|z_t1c − z_t1pre|⟩ in latent space.

    ``regions_latent`` are the brain / tumor / sub-region masks downsampled
    to the latent grid (typically 4× downsampling — see
    ``vena.data.h5.shared.mask_downsampler``). ``z_t1c`` and ``z_t1pre``
    have shape ``(1, C, h, w, d)``; the metrics are reduced across the
    channel dimension before masking.
    """
    if z_t1c.shape != z_t1pre.shape:
        raise ValueError(
            f"latent_space_contrast: shape mismatch — z_t1c {tuple(z_t1c.shape)} "
            f"vs z_t1pre {tuple(z_t1pre.shape)}"
        )
    # Reduce across channel: ⟨|z|⟩ over channels then per-voxel.
    abs_t1c = z_t1c.abs().mean(dim=1, keepdim=True)  # (1, 1, h, w, d)
    abs_t1pre = z_t1pre.abs().mean(dim=1, keepdim=True)
    abs_delta = (z_t1c - z_t1pre).abs().mean(dim=1, keepdim=True)

    out: dict[str, float] = {}
    for name, mask in [
        ("et", regions_latent.et),
        ("netc", regions_latent.netc),
        ("ed", regions_latent.ed),
        ("bnwt", regions_latent.bnwt),
        ("wt", regions_latent.wt),
    ]:
        out[f"latent_mean_abs_t1c_{name}"] = _masked_mean(abs_t1c, mask)
        out[f"latent_mean_abs_t1pre_{name}"] = _masked_mean(abs_t1pre, mask)
        out[f"latent_mean_abs_delta_{name}"] = _masked_mean(abs_delta, mask)
    return out


def signal_ratios(
    contrast_dict: dict[str, float],
    *,
    image: bool,
) -> dict[str, float]:
    """Compute the ET/BNWT signal ratios from a contrast dict.

    Parameters
    ----------
    contrast_dict : dict
        Output of either :func:`image_space_contrast` or
        :func:`latent_space_contrast`.
    image : bool
        ``True`` for image-space ratios (C4), ``False`` for latent (C5).
    """
    prefix = "image_mean_abs_diff" if image else "latent_mean_abs_delta"
    et = contrast_dict.get(f"{prefix}_et", float("nan"))
    bnwt = contrast_dict.get(f"{prefix}_bnwt", float("nan"))
    ratio_key = "image_signal_ratio_et_over_bnwt" if image else "latent_signal_ratio_et_over_bnwt"
    if bnwt is None or bnwt != bnwt or bnwt == 0:  # NaN or zero
        return {ratio_key: float("nan")}
    return {ratio_key: et / bnwt}


__all__ = [
    "image_space_contrast",
    "latent_space_contrast",
    "signal_ratios",
]

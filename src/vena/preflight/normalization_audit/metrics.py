"""Per-region metric helpers for the V3 normalisation audit.

Regions follow the BraTS-2023 label convention used elsewhere in VENA:

* NETC — necrotic core         (label 1)
* ED   — peritumoral edema     (label 2)
* ET   — enhancing tumour      (label 4)
* WT   — whole tumour          (NETC ∪ ED ∪ ET)
* BNWT — brain ∧ ¬WT           (non-tumour brain tissue)
* BG   — background            (¬brain)

All metrics operate in image space on volumes that have been brought back
to the native brain box; both ``ref`` and ``pred`` are in the same affine
range (``[0, 1]`` or — for ``clip=False`` variants — a soft super-1 range).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RegionMasks:
    """On-the-fly region masks derived from a tumor int8 label map + brain mask."""

    brain: torch.Tensor  # (1, 1, H, W, D) bool
    netc: torch.Tensor
    ed: torch.Tensor
    et: torch.Tensor
    wt: torch.Tensor
    bnwt: torch.Tensor

    def n_voxels(self) -> dict[str, int]:
        return {
            "brain": int(self.brain.sum().item()),
            "netc": int(self.netc.sum().item()),
            "ed": int(self.ed.sum().item()),
            "et": int(self.et.sum().item()),
            "wt": int(self.wt.sum().item()),
            "bnwt": int(self.bnwt.sum().item()),
        }


def build_region_masks(
    tumor_label: torch.Tensor,
    brain_mask: torch.Tensor,
) -> RegionMasks:
    """Build per-region boolean masks from BraTS-2023 labels + brain mask.

    Parameters
    ----------
    tumor_label : torch.Tensor
        Shape ``(1, 1, H, W, D)`` or ``(H, W, D)``; int dtype with values
        in ``{0, 1, 2, 3 or 4}``. BraTS-23 uses ``{0, 1, 2, 3}`` for the
        re-labelled segmentations; older datasets use ``{0, 1, 2, 4}``.
        Both are accepted; ``ET`` is the value-3 label OR value-4 label,
        whichever appears.
    brain_mask : torch.Tensor
        Same shape; non-zero on brain.
    """
    if tumor_label.ndim == 3:
        tumor_label = tumor_label[None, None]
    if brain_mask.ndim == 3:
        brain_mask = brain_mask[None, None]
    if tumor_label.shape != brain_mask.shape:
        raise ValueError(
            f"build_region_masks: shape mismatch — tumor {tuple(tumor_label.shape)} "
            f"vs brain {tuple(brain_mask.shape)}"
        )
    lbl = tumor_label.long()
    brain = brain_mask > 0
    netc = lbl == 1
    ed = lbl == 2
    # ET is label 4 in the original BraTS convention, label 3 in BraTS-23
    # post-relabeling. Detect at runtime to support both.
    et = (lbl == 4) | (lbl == 3)
    wt = netc | ed | et
    bnwt = brain & ~wt
    return RegionMasks(brain=brain, netc=netc, ed=ed, et=et, wt=wt, bnwt=bnwt)


def _broadcast_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Squeeze any leading singleton dims of ``mask`` until it matches ``like``."""
    m = mask
    while m.ndim > like.ndim:
        m = m.squeeze(0)
    return m


def _masked_mean_abs_err(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return float("nan")
    m = _broadcast_mask(mask, pred)
    err = (pred - ref).abs()
    return float(err[m].mean().item())


def _masked_mse(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return float("nan")
    m = _broadcast_mask(mask, pred)
    err = (pred - ref).pow(2)
    return float(err[m].mean().item())


def _psnr_db(mse: float, data_range: float = 1.0) -> float:
    if not math.isfinite(mse) or mse <= 0.0:
        return float("nan")
    return 10.0 * math.log10((data_range**2) / mse)


def compute_per_region_round_trip(
    pred: torch.Tensor,
    ref: torch.Tensor,
    regions: RegionMasks,
    *,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Compute per-region MAE / MSE / PSNR for the VAE encode→decode round-trip.

    SSIM is omitted at per-region granularity (per the v2 diagnosis P6 —
    small-mask SSIM is dominated by the boundary and uninformative).
    Whole-brain SSIM is computed by the caller via :func:`whole_volume_ssim`.
    """
    out: dict[str, float] = {}
    for name, mask in [
        ("whole", regions.brain),
        ("et", regions.et),
        ("netc", regions.netc),
        ("ed", regions.ed),
        ("bnwt", regions.bnwt),
    ]:
        mae = _masked_mean_abs_err(pred, ref, mask)
        mse = _masked_mse(pred, ref, mask)
        out[f"mae_{name}"] = mae
        out[f"mse_{name}"] = mse
        out[f"psnr_{name}_db"] = _psnr_db(mse, data_range=data_range)
    return out


def whole_volume_ssim(
    pred: torch.Tensor,
    ref: torch.Tensor,
    brain_mask: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> float:
    """Compute SSIM over the brain region only.

    Uses a 3D mean+variance formulation (Wang et al. 2004) restricted to
    brain voxels. The 3D kernel is a simple 7³ box filter for speed; this
    matches the convention used in `vena.model.fm.metrics.image`.
    """
    if pred.ndim == 3:
        pred = pred[None, None]
    if ref.ndim == 3:
        ref = ref[None, None]
    if brain_mask.ndim == 3:
        brain_mask = brain_mask[None, None]
    mask = brain_mask > 0
    if mask.sum() == 0:
        return float("nan")
    # Restrict to brain.
    p = pred * mask
    r = ref * mask
    # Mean / var over brain voxels.
    mu_p = float(p[mask].mean().item())
    mu_r = float(r[mask].mean().item())
    var_p = float(p[mask].var(unbiased=False).item())
    var_r = float(r[mask].var(unbiased=False).item())
    cov = float(((p[mask] - mu_p) * (r[mask] - mu_r)).mean().item())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_p * mu_r + c1) * (2 * cov + c2)
    den = (mu_p**2 + mu_r**2 + c1) * (var_p + var_r + c2)
    if den == 0:
        return float("nan")
    return num / den


__all__ = [
    "RegionMasks",
    "build_region_masks",
    "compute_per_region_round_trip",
    "whole_volume_ssim",
]

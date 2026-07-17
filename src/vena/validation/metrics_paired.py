"""Paired voxel-wise fidelity metrics for Phase-2 benchmark (§4.2, §4.5, §4.7).

Implements the metric suite from validation_proposal.md:

- §4.2: MAE, RMSE, PSNR-3D, SSIM-3D, MS-SSIM-3D per region (brain, wt,
  bg_undilated).
- §4.5: inference cost is a pass-through from ScanSample metadata.
- §4.7: ZGD — z-gradient discontinuity ratio for inter-slice consistency.

SSIM treatment decision
-----------------------
``monai.metrics.regression.compute_ssim_and_cs`` returns the full spatial SSIM
map (valid convolution; shape ``(B,C,H−k+1,W−k+1,D−k+1)`` for window size k).
We compute this map once per scan and average it inside each region mask after
center-cropping the mask by ``k//2`` voxels on each edge to match the reduced
spatial extent.  This is the principled "SSIM over region R = mean SSIM-map
inside R" interpretation — see :func:`ssim_map_3d` and :func:`ssim_in_region`.

This replaces the degenerate mean-fill proxy in
``vena.model.fm.metrics.ImageMetrics.ssim`` (documented as unacceptable in
model-coding-standards.md rule 14 and SHARED_CONTRACTS §11 trap 7).  The same
treatment is applied identically to every method, making the comparison valid.

MS-SSIM treatment decision
--------------------------
``monai.metrics.regression.compute_ms_ssim`` reduces spatially and returns a
scalar per batch — no per-voxel map is available from the API.  We therefore
use the following approximations:

- ``brain``: full brain volume (always valid; both volumes are zero outside).
- ``wt``: WT bounding-box crop + ``bbox_margin`` voxels; NaN when any spatial
  dimension is < 90 (MONAI minimum for 4-level MS-SSIM with kernel_size=11).
- ``bg_undilated``: same value as ``brain`` (WT is typically <5% of brain
  volume; contamination is negligible).

These limitations are recorded in every ``decision.json`` produced by the
engine.

Formula provenance
------------------
MAE, RMSE, PSNR are implemented in numpy with formulas identical to
``vena.model.fm.metrics.ImageMetrics``.  The compute-in-torch approach of
ImageMetrics is correct for training-time use but introduces unnecessary
overhead when iterating 21 k volumes — numpy is cleaner here.  The private
``_psnr``/``_ssim`` helpers from ``src/vena/competitors/*/inference.py`` are
NOT used.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vena.validation.io import ScanSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Masked scalar metrics — numpy (no torch overhead for per-volume iteration)
# ---------------------------------------------------------------------------

_EPS = 1e-8  # epsilon for masked denominators and PSNR clamping


def _masked_mae(
    pred: np.ndarray,
    real: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean absolute error inside ``mask``.

    Parameters
    ----------
    pred, real :
        ``(H, W, D)`` float32 volumes in [0, 1].
    mask :
        ``(H, W, D)`` bool region mask.

    Returns
    -------
    float
        NaN when the mask is empty.
    """
    n = mask.sum()
    if n == 0:
        return float("nan")
    return float(np.abs(pred[mask] - real[mask]).mean())


def _masked_rmse(
    pred: np.ndarray,
    real: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Root-mean-square error inside ``mask``.  NaN when mask is empty."""
    n = mask.sum()
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - real[mask]) ** 2)))


def _masked_psnr(
    pred: np.ndarray,
    real: np.ndarray,
    mask: np.ndarray,
    *,
    data_range: float = 1.0,
) -> float:
    """Region-masked PSNR-3D.  NaN when mask is empty.

    Formula: ``10 * log10(data_range^2 / MSE)`` where MSE is computed only
    over in-mask voxels.  Identical to
    ``vena.model.fm.metrics.ImageMetrics._masked_psnr_3d``.
    """
    n = mask.sum()
    if n == 0:
        return float("nan")
    mse = float(np.mean((pred[mask] - real[mask]) ** 2))
    if mse < _EPS:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


# ---------------------------------------------------------------------------
# SSIM — principled map approach
# ---------------------------------------------------------------------------


def ssim_map_3d(
    pred: np.ndarray,
    real: np.ndarray,
    *,
    data_range: float = 1.0,
    window_size: int = 7,
    window_sigma: float = 1.5,
) -> np.ndarray:
    """Compute the 3-D spatial SSIM map via MONAI ``compute_ssim_and_cs``.

    Returns the full-image SSIM map (valid convolution) without any spatial
    reduction.  Callers use :func:`ssim_in_region` to average inside a mask.

    Parameters
    ----------
    pred, real :
        ``(H, W, D)`` float32 volumes in [0, 1].
    data_range :
        Fixed at 1.0 (SHARED_CONTRACTS §11 trap 6).
    window_size :
        Gaussian window size (default 7, matching ImageMetrics default).
    window_sigma :
        Gaussian window sigma.

    Returns
    -------
    np.ndarray
        ``(H−k+1, W−k+1, D−k+1)`` float32 SSIM values where
        k = ``window_size``.  The trim on each edge is ``window_size // 2``.
    """
    from monai.metrics.regression import compute_ssim_and_cs

    # (1, 1, H, W, D)
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    real_t = torch.from_numpy(real).unsqueeze(0).unsqueeze(0).float()

    ks = (window_size, window_size, window_size)
    sigma = (window_sigma, window_sigma, window_sigma)

    with torch.no_grad():
        ssim_full, _ = compute_ssim_and_cs(
            pred_t,
            real_t,
            spatial_dims=3,
            kernel_size=ks,
            kernel_sigma=sigma,
            data_range=data_range,
        )
    # ssim_full: (1, 1, H-k+1, W-k+1, D-k+1)
    return ssim_full.squeeze().numpy().astype(np.float32)


def ssim_in_region(
    ssim_map: np.ndarray,
    mask: np.ndarray,
    *,
    window_size: int = 7,
) -> float:
    """Average SSIM map values inside a region mask.

    The mask is center-cropped by ``window_size // 2`` voxels on each edge to
    match the spatial extent of ``ssim_map`` (which has reduced dimensions due
    to valid convolution).

    Parameters
    ----------
    ssim_map :
        ``(H−k+1, W−k+1, D−k+1)`` SSIM map from :func:`ssim_map_3d`.
    mask :
        ``(H, W, D)`` bool region mask (original spatial dims).
    window_size :
        Must match the value used to compute ``ssim_map``.

    Returns
    -------
    float
        Mean SSIM inside the region.  NaN when no cropped mask voxels are
        active (tiny region entirely within the trim margin).
    """
    trim = window_size // 2
    # Uppercase H/W/D is this codebase's shape convention (coding-standards
    # rule 11 writes shapes as Float["B C H W D"]); lowercasing would read as
    # scalars rather than volume extents.
    H, W, D = mask.shape  # noqa: N806
    # Center-crop: remove ``trim`` voxels from each side.
    mask_cropped: np.ndarray = mask[trim : H - trim, trim : W - trim, trim : D - trim]
    in_region = ssim_map[mask_cropped]
    if in_region.size == 0:
        return float("nan")
    return float(np.mean(in_region))


# ---------------------------------------------------------------------------
# MS-SSIM — brain-level scalar (API limitation documented)
# ---------------------------------------------------------------------------


def ms_ssim_brain(
    pred: np.ndarray,
    real: np.ndarray,
    brain: np.ndarray,
    *,
    weights: list[float] | tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.3633),
    data_range: float = 1.0,
) -> float:
    """MS-SSIM on the full brain volume.

    Both ``pred`` and ``real`` are zero outside the brain mask, so computing
    MS-SSIM on the full volume is equivalent to brain-restricted MS-SSIM.

    Parameters
    ----------
    pred, real :
        ``(H, W, D)`` float32 volumes — already zero outside brain.
    brain :
        ``(H, W, D)`` bool brain mask.  Used only to guard against empty
        brain (returns NaN).
    weights :
        Per-scale MS-SSIM weights.  Default: Wang 2003 4-scale weights
        ``[0.0448, 0.2856, 0.3001, 0.3633]``.
    data_range :
        Fixed at 1.0 (SHARED_CONTRACTS §11 trap 6).

    Returns
    -------
    float
        NaN when brain is empty or MONAI raises (volume too small).
    """
    if brain.sum() == 0:
        return float("nan")
    return _compute_ms_ssim_tensor(pred, real, weights=weights, data_range=data_range)


def ms_ssim_wt_bbox(
    pred: np.ndarray,
    real: np.ndarray,
    wt: np.ndarray,
    *,
    weights: list[float] | tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.3633),
    bbox_margin: int = 8,
    data_range: float = 1.0,
    min_dim: int = 90,
) -> float:
    """MS-SSIM on the WT bounding-box crop.

    When the cropped volume is too small for the specified number of MS-SSIM
    levels (any dim < ``min_dim``), returns NaN rather than raising.

    Parameters
    ----------
    pred, real :
        ``(H, W, D)`` float32 volumes.
    wt :
        ``(H, W, D)`` bool WT mask.
    bbox_margin :
        Extra voxels added to each side of the WT bounding box.
    data_range :
        Fixed at 1.0.
    min_dim :
        Minimum voxels in any spatial dim required for 4-level MS-SSIM with
        kernel_size=11.  Volumes below this threshold return NaN.

    Returns
    -------
    float
        NaN when WT is empty or the bbox is too small.
    """
    if wt.sum() == 0:
        return float("nan")

    # Bounding box
    coords = np.argwhere(wt)
    lo = coords.min(axis=0) - bbox_margin
    hi = coords.max(axis=0) + bbox_margin + 1  # exclusive
    H, W, D = wt.shape  # noqa: N806  — shape convention, see above
    lo = np.clip(lo, 0, [H, W, D])
    hi = np.clip(hi, 0, [H, W, D])
    shape = hi - lo

    if (shape < min_dim).any():
        logger.debug(
            "ms_ssim_wt_bbox: WT bbox %s < min_dim=%d, returning NaN", shape.tolist(), min_dim
        )
        return float("nan")

    pred_crop = pred[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    real_crop = real[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    return _compute_ms_ssim_tensor(pred_crop, real_crop, weights=weights, data_range=data_range)


def _compute_ms_ssim_tensor(
    pred: np.ndarray,
    real: np.ndarray,
    *,
    weights: list[float] | tuple[float, ...],
    data_range: float,
) -> float:
    """Internal: convert to tensor and call MONAI compute_ms_ssim."""
    from monai.metrics.regression import compute_ms_ssim

    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    real_t = torch.from_numpy(real).unsqueeze(0).unsqueeze(0).float()
    try:
        with torch.no_grad():
            val = compute_ms_ssim(
                pred_t,
                real_t,
                spatial_dims=3,
                data_range=data_range,
                weights=list(weights),
            )
        return float(val.item())
    except (RuntimeError, ValueError) as exc:
        logger.debug("compute_ms_ssim failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# ZGD — z-gradient discontinuity ratio (§4.7)
# ---------------------------------------------------------------------------


def zgd(
    pred: np.ndarray,
    real: np.ndarray,
    brain: np.ndarray,
) -> float:
    """Z-gradient discontinuity ratio.

    Measures inter-slice consistency relative to the real T1c.

    Formula::

        mean|∂z I| = E_{(x,y,z) ∈ brain[...,:-1]} |I[x,y,z+1] − I[x,y,z]|
        ZGD        = mean|∂z pred| / mean|∂z real|

    A score > 1 indicates more inter-slice variation than the real (the 2-D
    tier's slice-stacking artefact).  ≈ 1 is ideal; < 1 means over-smoothed
    in z.  Expected: ≈ 1 for 3-D-native methods, > 1 for C1/C2/C3.

    Parameters
    ----------
    pred, real :
        ``(H, W, D)`` float32 volumes.
    brain :
        ``(H, W, D)`` bool brain mask.

    Returns
    -------
    float
        NaN when the brain mask has fewer than two z-slices.
    """
    # Gradient positions: voxels at z=0..D-2 that are in the brain mask
    brain_src: np.ndarray = brain[..., :-1]  # (H, W, D-1)
    n = brain_src.sum()
    if n == 0:
        return float("nan")

    pred_grad = np.abs(pred[..., 1:] - pred[..., :-1])  # (H, W, D-1)
    real_grad = np.abs(real[..., 1:] - real[..., :-1])

    pred_mean = float(pred_grad[brain_src].mean())
    real_mean = float(real_grad[brain_src].mean())

    if real_mean < _EPS:
        return float("nan")
    return pred_mean / real_mean


# ---------------------------------------------------------------------------
# ScanMetrics dataclass — one row of the per_scan CSV
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanMetrics:
    """Per-scan metric bundle for one (method, cohort, nfe, scan_id) tuple.

    All float fields are NaN when the corresponding region is empty or the
    computation failed (never silently dropped — see §4.2 and task spec §7).
    """

    # -- Identity --
    scan_id: str
    patient_id: str
    method: str
    cohort: str
    ring: str
    nfe: int

    # -- §4.5 inference cost (pass-through from ScanSample) --
    inference_seconds: float
    peak_vram_mb: float

    # -- §4.1 scoring-space audit (pass-through from ScanSample / select_scoring_volume) --
    pred_mode: str  # "raw" | "harmonised" — which volume was scored
    raw_p995: float  # brain p99.5 of the raw prediction (under-saturation audit)

    # -- §4.2 MAE × 3 regions --
    mae_brain: float
    mae_wt: float
    mae_bg_undilated: float

    # -- §4.2 RMSE × 3 regions --
    rmse_brain: float
    rmse_wt: float
    rmse_bg_undilated: float

    # -- §4.2 PSNR × 3 regions --
    psnr_brain: float
    psnr_wt: float
    psnr_bg_undilated: float

    # -- §4.2 SSIM × 3 regions (principled: SSIM-map average inside region) --
    ssim_brain: float
    ssim_wt: float
    ssim_bg_undilated: float

    # -- §4.2 MS-SSIM × 3 regions (see module-level treatment note) --
    ms_ssim_brain: float
    ms_ssim_wt: float  # WT bbox crop; NaN when bbox < 90 in any dim
    ms_ssim_bg_undilated: float  # = ms_ssim_brain (API limitation, documented)

    # -- §4.7 ZGD --
    zgd: float

    # -- Diagnostics --
    n_brain_voxels: int
    n_wt_voxels: int
    n_bg_undilated_voxels: int

    def to_flat_dict(self) -> dict[str, object]:
        """Return a flat dict suitable for a DataFrame row."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Top-level compute function
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricConfig:
    """Metric computation parameters (injected from YAML config)."""

    data_range: float = 1.0
    ssim_window_size: int = 7
    ssim_window_sigma: float = 1.5
    ms_ssim_weights: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.3633)
    ms_ssim_bbox_margin: int = 8
    dilate_k: int = 5


def compute_paired_metrics(
    scan: ScanSample,
    cfg: MetricConfig,
) -> ScanMetrics:
    """Compute all paired-fidelity metrics for one scan.

    Parameters
    ----------
    scan :
        One scan from :func:`vena.validation.io.iter_scans`.
    cfg :
        Metric configuration (data_range, window sizes, etc.).

    Returns
    -------
    ScanMetrics
        All metrics for this (method, cohort, nfe, scan_id) tuple.
        Regions without voxels produce NaN scalars, not exceptions.
    """
    from vena.validation.regions import region_masks

    pred = scan.pred  # (H, W, D) float32
    real = scan.real  # (H, W, D) float32

    # Build region masks from brain/wt
    regions = region_masks(scan.brain, scan.wt, dilate_k=cfg.dilate_k)
    brain_mask: np.ndarray = regions["brain"]
    wt_mask: np.ndarray = regions["wt"]
    bg_mask: np.ndarray = regions["bg_undilated"]  # §4.2 uses bg_undilated

    # Count voxels for diagnostics
    n_brain = int(brain_mask.sum())
    n_wt = int(wt_mask.sum())
    n_bg = int(bg_mask.sum())

    # ---- MAE ----
    mae_brain = _masked_mae(pred, real, brain_mask)
    mae_wt = _masked_mae(pred, real, wt_mask)
    mae_bg = _masked_mae(pred, real, bg_mask)

    # ---- RMSE ----
    rmse_brain = _masked_rmse(pred, real, brain_mask)
    rmse_wt = _masked_rmse(pred, real, wt_mask)
    rmse_bg = _masked_rmse(pred, real, bg_mask)

    # ---- PSNR ----
    psnr_brain = _masked_psnr(pred, real, brain_mask, data_range=cfg.data_range)
    psnr_wt = _masked_psnr(pred, real, wt_mask, data_range=cfg.data_range)
    psnr_bg = _masked_psnr(pred, real, bg_mask, data_range=cfg.data_range)

    # ---- SSIM (principled map approach) ----
    # Compute the full SSIM map once and average inside each region.
    _ssim_map = ssim_map_3d(
        pred,
        real,
        data_range=cfg.data_range,
        window_size=cfg.ssim_window_size,
        window_sigma=cfg.ssim_window_sigma,
    )
    ssim_brain = ssim_in_region(_ssim_map, brain_mask, window_size=cfg.ssim_window_size)
    ssim_wt = ssim_in_region(_ssim_map, wt_mask, window_size=cfg.ssim_window_size)
    ssim_bg = ssim_in_region(_ssim_map, bg_mask, window_size=cfg.ssim_window_size)

    # ---- MS-SSIM (API limitation: no spatial map) ----
    _ms_brain = ms_ssim_brain(
        pred,
        real,
        brain_mask,
        weights=cfg.ms_ssim_weights,
        data_range=cfg.data_range,
    )
    _ms_wt = ms_ssim_wt_bbox(
        pred,
        real,
        wt_mask,
        weights=cfg.ms_ssim_weights,
        bbox_margin=cfg.ms_ssim_bbox_margin,
        data_range=cfg.data_range,
    )
    # bg_undilated: same as brain (documented — WT <5% of brain volume)
    _ms_bg = _ms_brain

    # ---- ZGD ----
    _zgd = zgd(pred, real, brain_mask)

    return ScanMetrics(
        scan_id=scan.scan_id,
        patient_id=scan.patient_id,
        method=scan.method,
        cohort=scan.cohort,
        ring=scan.ring,
        nfe=scan.nfe,
        inference_seconds=scan.inference_seconds,
        peak_vram_mb=scan.peak_vram_mb,
        pred_mode=scan.pred_mode,
        raw_p995=scan.raw_p995,
        mae_brain=mae_brain,
        mae_wt=mae_wt,
        mae_bg_undilated=mae_bg,
        rmse_brain=rmse_brain,
        rmse_wt=rmse_wt,
        rmse_bg_undilated=rmse_bg,
        psnr_brain=psnr_brain,
        psnr_wt=psnr_wt,
        psnr_bg_undilated=psnr_bg,
        ssim_brain=ssim_brain,
        ssim_wt=ssim_wt,
        ssim_bg_undilated=ssim_bg,
        ms_ssim_brain=_ms_brain,
        ms_ssim_wt=_ms_wt,
        ms_ssim_bg_undilated=_ms_bg,
        zgd=_zgd,
        n_brain_voxels=n_brain,
        n_wt_voxels=n_wt,
        n_bg_undilated_voxels=n_bg,
    )

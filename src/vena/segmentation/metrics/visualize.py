"""Pure, headless figure builders for soft-mask QC and injection sanity.

All functions write a PNG to *path* and return it.  No display is produced —
matplotlib backend is forced to ``Agg`` at module import so this module is
safe to import on any headless server or in pytest without a display.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # force headless — must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

from vena.data.h5.latent_domain.manifest import LATENT_SPATIAL
from vena.segmentation.exceptions import SegMetricError

if TYPE_CHECKING:
    import pandas as pd

    # CropPadSpec lives in vena.common (MAISI adapter). Importing at module
    # level would drag in all MAISI model code; defer to the call site.
    from vena.common import CropPadSpec

__all__ = [
    "PatientView",
    "check_mask_invariants",
    "compute_mask_stats",
    "compute_residual_energy_ratio",
    "render_injection_sanity",
    "render_latent_embedding",
    "render_mask_qc",
    "render_slice_montage",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EMPTY_MASK_THRESHOLD: float = 0.01
_NETC_VIOLATION_EPSILON: float = 1e-6

# High-contrast overlay colours: TC = saturated green; NETC = saturated magenta.
# Used for hard-mask (binary) rows and as contour colours on soft rows.
_WT_COLOR: tuple[float, float, float] = (0.1, 0.9, 0.2)
_NETC_COLOR: tuple[float, float, float] = (1.0, 0.1, 0.6)
_COLORS: tuple[tuple[float, float, float], ...] = (_WT_COLOR, _NETC_COLOR)
_CONTOUR_LEVELS: list[float] = [0.25, 0.5, 0.75]
_CONTOUR_LW: float = 0.6

# Perceptual colormaps for continuous soft-probability overlays.
# Higher probability → hotter/darker colour.  Both are visually distinct on
# greyscale MRI and map naturally onto the existing green/magenta convention.
_TC_CMAP = plt.cm.YlGn  # yellow (low prob) → dark green (high prob)  — TC
_NETC_CMAP = plt.cm.RdPu  # light pink (low) → dark magenta (high)       — NETC
_CMAPS: tuple[Any, ...] = (_TC_CMAP, _NETC_CMAP)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch Tensor or numpy array to numpy, no-op otherwise."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _decode_ids(raw: np.ndarray) -> list[str]:
    """Decode a vlen-str or bytes array of scan IDs to a list of str."""
    return [id_.decode() if isinstance(id_, bytes) else str(id_) for id_ in raw]


def _axial_tumor_slices(soft_mask: np.ndarray, n_cols: int) -> np.ndarray:
    """Return *n_cols* evenly-spaced depth indices covering the tumour extent.

    Parameters
    ----------
    soft_mask : np.ndarray
        Shape ``(2, H, W, D)``.  Channel 0 = TC (default) or WT (ablation).
    n_cols : int
        Number of slices to return.

    Returns
    -------
    np.ndarray
        1-D int array of length *n_cols*, depth indices into axis 2.
    """
    ch0 = soft_mask[0]  # (H, W, D) — TC or WT depending on cfg.tumor_region
    depth_presence = ch0.sum(axis=(0, 1)) > 0  # area-based: any tumour-covered voxels
    z_tumour = np.where(depth_presence)[0]
    if len(z_tumour) == 0:
        d = ch0.shape[2]
        return np.linspace(0, d - 1, n_cols, dtype=int)
    return np.linspace(z_tumour[0], z_tumour[-1], n_cols, dtype=int)


def _overlay_rgba(
    soft_ch: np.ndarray,
    color_rgb: tuple[float, float, float],
    alpha_max: float,
) -> np.ndarray:
    """Build a (H, W, 4) RGBA overlay; opacity proportional to probability.

    Parameters
    ----------
    soft_ch : np.ndarray
        2-D float array in ``[0, 1]``, shape ``(H, W)``.
    color_rgb : tuple[float, float, float]
        Fixed RGB colour for the overlay (R, G, B in ``[0, 1]``).
    alpha_max : float
        Maximum overlay opacity; per-pixel alpha = ``soft_ch * alpha_max``.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 4)`` float32 RGBA image.
    """
    h, w = soft_ch.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = np.clip(soft_ch * alpha_max, 0.0, 1.0)
    return rgba


def _overlay_cmap_rgba(
    soft_ch: np.ndarray,
    cmap: Any,
    alpha_max: float,
) -> np.ndarray:
    """Build (H, W, 4) RGBA using a perceptual colormap; alpha ∝ probability.

    Higher probability → hotter/darker colour **and** higher opacity.  At
    probability 0 the overlay is fully transparent so the anatomy layer shows.

    Parameters
    ----------
    soft_ch : np.ndarray
        2-D float array in ``[0, 1]``, shape ``(H, W)``.
    cmap : callable
        A matplotlib colormap (e.g. ``plt.cm.YlGn``, ``plt.cm.RdPu``).
        Called as ``cmap(array)`` → ``(H, W, 4)`` RGBA.
    alpha_max : float
        Maximum overlay opacity for pixels at probability 1.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 4)`` float32 RGBA.
    """
    rgba = cmap(np.clip(soft_ch, 0.0, 1.0)).astype(np.float32)  # (H, W, 4)
    # Replace cmap's alpha with probability-proportional opacity so background
    # anatomy remains visible and the halo (low-prob boundary) is semi-transparent.
    rgba[:, :, 3] = np.clip(soft_ch * alpha_max, 0.0, 1.0)
    return rgba


def _add_contours(
    ax: Any,
    arr: np.ndarray,
    color_rgb: tuple[float, float, float],
) -> None:
    """Draw probability contour lines at levels 0.25, 0.5, 0.75.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes; must already contain the base imshow call.
    arr : np.ndarray
        2-D float array in ``[0, 1]``, same spatial orientation as the imshow.
    color_rgb : tuple[float, float, float]
        Contour colour — same hue as the filled overlay.
    """
    if arr.max() <= 0.0:
        return  # nothing to contour; avoids empty-contour matplotlib warning
    ax.contour(arr, levels=_CONTOUR_LEVELS, colors=[color_rgb], linewidths=_CONTOUR_LW, alpha=0.9)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class PatientView:
    """Per-patient data bundle for montage rendering.

    Attributes
    ----------
    patient_id : str
        Identifier shown in figure annotations.
    t1pre : np.ndarray
        T1pre volume, shape ``(H, W, D)``, float in ``[0, 1]``.
    soft_mask : np.ndarray
        Soft ``[TC, NETC]`` (or ``[WT, NETC]`` for ablation) probability map
        at **image** resolution, shape ``(2, H, W, D)``, float in ``[0, 1]``.
        Engine sets this to the true sigmoid(SDT/σ) map produced by
        :func:`vena.segmentation.targets.soft_targets.make_soft_targets`.
    tumor_volume : float
        Tumour volume in voxels (WT channel sum); used for row ordering.
    cohort : str
        Cohort tag for colour-coding in the embedding figure.
    """

    patient_id: str
    t1pre: np.ndarray
    soft_mask: np.ndarray
    tumor_volume: float
    cohort: str = field(default="")
    # Integer BraTS-style tumour label (H,W,D); used by render_mask_qc for the
    # hard-mask row so WT=(label>0) and NETC=(label==1) render correctly.
    # None is the legacy default — callers should always provide the true label.
    hard_label: np.ndarray | None = field(default=None)


# ---------------------------------------------------------------------------
# Machine-stats helpers
# ---------------------------------------------------------------------------


def compute_mask_stats(soft_masks: np.ndarray) -> dict[str, float | int]:
    """Compute machine stats from a batch of soft masks.

    Parameters
    ----------
    soft_masks : np.ndarray
        Float32 array of shape ``(N, 2, H, W, D)`` in ``[0, 1]``.
        Channel 0 = TC or WT (per cfg.tumor_region), channel 1 = NETC.

    Returns
    -------
    dict
        ``soft_mass_fraction_in_wt`` (float): fraction of total soft
        probability mass that lies within the binarized WT region
        (``WT > 0.5``).
        ``netc_violation_count`` (int): total number of voxels across all
        patients where ``NETC > WT + epsilon`` (nesting violated).
        ``empty_mask_count`` (int): number of patients whose WT channel
        maximum falls below :data:`_EMPTY_MASK_THRESHOLD`.

    Raises
    ------
    SegMetricError
        If *soft_masks* has fewer than 2 channels in axis 1 or is not 5-D.
    """
    if soft_masks.ndim != 5 or soft_masks.shape[1] < 2:
        raise SegMetricError(f"soft_masks must be (N, 2, H, W, D); got {soft_masks.shape}")
    wt = soft_masks[:, 0]  # (N, H, W, D)
    netc = soft_masks[:, 1]  # (N, H, W, D)

    # Soft-mass fraction in WT: fraction of total mass inside binarized WT
    wt_binary = (wt > 0.5).astype(np.float32)
    total_mass = float(wt.sum() + netc.sum())
    if total_mass > 0.0:
        in_wt_mass = float((wt * wt_binary).sum() + (netc * wt_binary).sum())
        soft_mass_fraction_in_wt = in_wt_mass / total_mass
    else:
        soft_mass_fraction_in_wt = 0.0

    # NETC⊆WT violation: voxels where NETC > WT + epsilon
    netc_violation_count = int((netc > wt + _NETC_VIOLATION_EPSILON).sum())

    # Empty-mask count: patients where max WT < threshold
    per_patient_wt_max = wt.reshape(wt.shape[0], -1).max(axis=1)  # (N,)
    empty_mask_count = int((per_patient_wt_max < _EMPTY_MASK_THRESHOLD).sum())

    return {
        "soft_mass_fraction_in_wt": soft_mass_fraction_in_wt,
        "netc_violation_count": netc_violation_count,
        "empty_mask_count": empty_mask_count,
    }


def compute_residual_energy_ratio(
    residuals: np.ndarray,
    wt_mask: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> float:
    """Compute the in-WT to out-of-WT residual-energy ratio.

    Parameters
    ----------
    residuals : np.ndarray
        Residual energy map, shape ``(H, W, D)`` or ``(C, H, W, D)``.
        Interpreted as squared L2 per-voxel energy.
    wt_mask : np.ndarray
        Binary or soft WT mask, shape ``(H, W, D)`` or ``(2, H, W, D)``.
        Channel 0 is used when 4-D.
    epsilon : float
        Denominator guard for the out-of-WT energy.

    Returns
    -------
    float
        ``E_in_WT / (E_out_WT + epsilon)``; > 1 means the residual is
        concentrated inside the tumour region.
    """
    res = np.asarray(residuals, dtype=np.float32)
    wt = np.asarray(wt_mask, dtype=np.float32)
    if res.ndim == 4:
        res = res.sum(axis=0)  # collapse channel dim
    if wt.ndim == 4:
        wt = wt[0]  # use WT channel

    wt_bin = (wt > 0.5).astype(np.float32)
    energy = res**2
    e_in = float((energy * wt_bin).sum())
    e_out = float((energy * (1.0 - wt_bin)).sum())
    return e_in / (e_out + epsilon)


# ---------------------------------------------------------------------------
# Invariant checker
# ---------------------------------------------------------------------------


def check_mask_invariants(
    soft_img_crop: np.ndarray,
    hard_label_crop: np.ndarray,
    soft_latent: np.ndarray,
    *,
    patient_id: str,
) -> dict[str, Any]:
    """Verify per-patient mask invariants; return a stats dict.

    All three inputs must be in the **crop frame** (192, 224, 192).
    Specifically:

    * *soft_img_crop* is the sigmoid(SDT/σ) probability map already
      cropped to ``(2, 192, 224, 192)`` via :func:`vena.common.apply_crop_pad`.
    * *hard_label_crop* is the BraTS integer label cropped to ``(192, 224, 192)``.
    * *soft_latent* is the latent-grid soft mask ``(2, 48, 56, 48)`` which
      this function up-scales ×4 → ``(2, 192, 224, 192)`` via trilinear
      interpolation before comparison.

    Invariants checked
    ------------------
    (a) **hard ⊆ soft**: every hard-TC voxel has ``soft_TC > 0.5`` (the
        calibration guarantee: soft > 0.5 ↔ hard label).  Reports the
        fraction of hard-TC voxels that violate this (should be ≈ 0).
    (b) **soft continuous**: a non-trivial fraction of TC-channel voxels
        lies in ``(0.05, 0.95)`` (the halo).  A value near 0 means the
        soft map has collapsed to binary — SDT derivation failure.
    (c) **latent ≈ image (registration)**: IoU of the up-scaled latent
        ``> 0.5`` region and the image-soft ``> 0.5`` region.  Low IoU
        (≲ 0.6) or large centroid distance indicates a crop/pool
        registration bug in ``pool_to_latent`` — **do not fix silently**.

    Parameters
    ----------
    soft_img_crop : np.ndarray
        Shape ``(2, 192, 224, 192)``, float32 in ``[0, 1]``.
        Channel 0 = TC/WT, channel 1 = NETC.
    hard_label_crop : np.ndarray
        Shape ``(192, 224, 192)``, int32 (BraTS-2021 convention).
    soft_latent : np.ndarray
        Shape ``(2, 48, 56, 48)``, float32 in ``[0, 1]``.
    patient_id : str
        Used for log messages; included in the returned dict.

    Returns
    -------
    dict
        Keys:

        * ``patient_id`` (str)
        * ``hard_subset_soft_violation_frac`` (float) — (a)
        * ``soft_intermediate_frac`` (float) — (b)
        * ``latent_image_iou`` (float) — (c)
        * ``latent_image_centroid_dist_vox`` (float | None) — (c)
        * ``invariant_ok`` (bool) — True when all checks pass

    Raises
    ------
    SegMetricError
        If array shapes are inconsistent.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    # Validate shapes
    expected_crop = (2, 192, 224, 192)
    if soft_img_crop.shape != expected_crop:
        raise SegMetricError(f"soft_img_crop must be {expected_crop}; got {soft_img_crop.shape}")
    if hard_label_crop.shape != (192, 224, 192):
        raise SegMetricError(
            f"hard_label_crop must be (192, 224, 192); got {hard_label_crop.shape}"
        )
    expected_lat = (2, *LATENT_SPATIAL)
    if soft_latent.shape != expected_lat:
        raise SegMetricError(f"soft_latent must be {expected_lat}; got {soft_latent.shape}")

    # Up-scale latent ×4 → (2, 192, 224, 192) via trilinear interpolation.
    # avg-pool is a box filter; trilinear upscale is the approximate inverse.
    lat_t = torch.from_numpy(soft_latent).float().unsqueeze(0)  # (1, 2, 48, 56, 48)
    lat_up = F.interpolate(lat_t, scale_factor=4, mode="trilinear", align_corners=False)
    lat_up_np: np.ndarray = lat_up.squeeze(0).numpy()  # (2, 192, 224, 192)

    soft_tc = soft_img_crop[0]  # (192, 224, 192)

    # (a) hard ⊆ soft: TC hard = (label > 0) & (label != 2)
    tc_hard = (hard_label_crop > 0) & (hard_label_crop != 2)
    n_hard = int(tc_hard.sum())
    if n_hard > 0:
        n_violation = int((soft_tc[tc_hard] <= 0.5).sum())
        hard_subset_soft_violation_frac = float(n_violation) / n_hard
    else:
        hard_subset_soft_violation_frac = 0.0

    # (b) soft continuous: fraction of voxels in (0.05, 0.95)
    n_vox = int(soft_tc.size)
    n_intermediate = int(((soft_tc > 0.05) & (soft_tc < 0.95)).sum())
    soft_intermediate_frac = float(n_intermediate) / n_vox if n_vox > 0 else 0.0

    # (c) registration IoU: upscaled-latent vs image-soft, both >0.5
    lat_up_tc = lat_up_np[0]  # (192, 224, 192)
    lat_bin = lat_up_tc > 0.5
    soft_bin = soft_tc > 0.5
    intersection = int((lat_bin & soft_bin).sum())
    union = int((lat_bin | soft_bin).sum())
    iou = float(intersection) / float(union + 1e-8) if union > 0 else 0.0

    # Centroid distance in voxels (None when either region is empty)
    if soft_bin.any() and lat_bin.any():
        c_soft = np.array(np.where(soft_bin)).mean(axis=1)  # (3,)
        c_lat = np.array(np.where(lat_bin)).mean(axis=1)  # (3,)
        centroid_dist: float | None = float(np.linalg.norm(c_soft - c_lat))
    else:
        centroid_dist = None

    # Whether any TC region is present — gates checks (b) and (c).
    # Patients with no TC (e.g. edema-only, label==2 throughout) are valid:
    # make_soft_targets returns all-zero TC channel for empty TC sets, which
    # produces soft_cont=0 and IoU=0 by construction — not a derivation bug.
    has_tc_region = (n_hard > 0) or (float(soft_tc.max()) > 0.05)

    # Threshold judgements — thresholds are intentionally conservative
    _hard_ok = hard_subset_soft_violation_frac < 0.05
    # (b) and (c) only apply when TC region exists; empty-TC patients trivially pass
    _soft_ok = (soft_intermediate_frac > 0.001) if has_tc_region else True
    _reg_ok = (
        iou >= 0.60 and (centroid_dist is None or centroid_dist < 20.0) if has_tc_region else True
    )
    invariant_ok = _hard_ok and _soft_ok and _reg_ok

    if not invariant_ok:
        logger.warning(
            "check_mask_invariants FAIL %s | hard_viol=%.3f  soft_cont=%.4f  IoU=%.3f  "
            "centroid=%.1f  has_tc=%s  [hard_ok=%s soft_ok=%s reg_ok=%s]",
            patient_id,
            hard_subset_soft_violation_frac,
            soft_intermediate_frac,
            iou,
            centroid_dist if centroid_dist is not None else float("nan"),
            has_tc_region,
            _hard_ok,
            _soft_ok,
            _reg_ok,
        )

    return {
        "patient_id": patient_id,
        "has_tc_region": has_tc_region,
        "hard_subset_soft_violation_frac": hard_subset_soft_violation_frac,
        "soft_intermediate_frac": soft_intermediate_frac,
        "latent_image_iou": iou,
        "latent_image_centroid_dist_vox": centroid_dist,
        "invariant_ok": invariant_ok,
    }


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def render_mask_qc(
    image: np.ndarray,
    hard_mask: np.ndarray,
    soft_mask_img: np.ndarray,
    soft_mask_latent: np.ndarray,
    *,
    patient_id: str,
    path: Path,
    roi_label: str = "TC",
    crop_spec: CropPadSpec | None = None,
) -> Path:
    """Produce a 3-row QC figure for a single patient.

    Row 0: T1pre anatomy + hard-mask overlay (``roi_label`` | NETC).
    Row 1: T1pre anatomy + soft-mask overlay at image resolution with
           perceptual colormap and probability contours (``roi_label`` | NETC).
    Row 2: T1pre anatomy + soft-mask at the latent grid, upscaled ×4 and
           rendered with the same colormap + contours (``roi_label`` | NETC).

    When *crop_spec* is provided (recommended):

    * All three arrays are projected into the **crop frame** ``(192, 224, 192)``
      via :func:`vena.common.apply_crop_pad` (image, hard mask, soft image)
      and trilinear upscaling ×4 (latent → crop frame).
    * A **single reference depth slice** ``k`` is chosen as the argmax of
      the hard-TC area in the crop frame so that ALL rows show the same
      physical z-level.  This eliminates the independent-argmax discrepancy
      that made masks appear in one row but not another.
    * Row 2 shows anatomy underneath the latent-upscaled overlay (same frame
      as Rows 0/1) rather than a plain black background.

    When *crop_spec* is ``None`` (legacy/test path):

    * Slice selection is independent per resolution (old behaviour) and
      rows may not align physically.

    Parameters
    ----------
    image : np.ndarray
        T1pre volume, shape ``(H, W, D)``, float in ``[0, 1]``.
    hard_mask : np.ndarray
        Integer label map ``(H, W, D)`` (BraTS convention); or
        ``(2, H, W, D)`` pre-binarized per-channel.
    soft_mask_img : np.ndarray
        Soft ``[roi_label, NETC]`` map at image resolution, shape
        ``(2, H, W, D)``.  Engine should provide the true sigmoid(SDT/σ)
        result from :func:`make_soft_targets`, not a binary approximation.
    soft_mask_latent : np.ndarray
        Soft ``[roi_label, NETC]`` map at latent grid, shape
        ``(2, *LATENT_SPATIAL)`` = ``(2, 48, 56, 48)``.
    patient_id : str
        Label used in the figure suptitle.
    path : Path
        Output PNG path; parent directories must exist or will be created.
    roi_label : str
        Human-readable name for channel 0 (default ``"TC"`` = tumour core).
        Use ``"WT"`` for legacy whole-tumour ablation runs.  Affects panel
        titles and hard-mask rendering: ``"TC"`` uses
        ``(label > 0) & (label != 2)``; any other value uses ``label > 0``.
    crop_spec : CropPadSpec or None
        Per-scan brain-centred crop specification from the image H5.  When
        provided, all arrays are aligned to the crop frame before rendering
        (recommended).  When ``None``, falls back to the legacy independent-
        slice behaviour (backward-compatible, used by tests without a crop_spec).

    Returns
    -------
    Path
        *path* after writing.

    Raises
    ------
    SegMetricError
        If *soft_mask_latent* shape does not match ``(2, *LATENT_SPATIAL)``.
    """
    expected_lat = (2, *LATENT_SPATIAL)
    if soft_mask_latent.shape != expected_lat:
        raise SegMetricError(
            f"soft_mask_latent must be {expected_lat}; got {soft_mask_latent.shape}"
        )

    # ------------------------------------------------------------------
    # Branch A: consistent crop-frame rendering (crop_spec provided)
    # ------------------------------------------------------------------
    if crop_spec is not None:
        return _render_mask_qc_crop_frame(
            image=image,
            hard_mask=hard_mask,
            soft_mask_img=soft_mask_img,
            soft_mask_latent=soft_mask_latent,
            patient_id=patient_id,
            path=path,
            roi_label=roi_label,
            crop_spec=crop_spec,
        )

    # ------------------------------------------------------------------
    # Branch B: legacy independent-slice rendering (no crop_spec)
    # ------------------------------------------------------------------
    # Pick the axial slice with the largest channel-0 tumour area (sum over H, W)
    ch0_img = soft_mask_img[0]  # (H, W, D)
    depth_sums_img = ch0_img.sum(axis=(0, 1))
    k_img = int(np.argmax(depth_sums_img)) if depth_sums_img.max() > 0 else ch0_img.shape[2] // 2

    # Same area criterion for the latent-grid row (depth = axis 2 of LATENT_SPATIAL)
    ch0_lat = soft_mask_latent[0]  # (48, 56, 48)
    depth_sums_lat = ch0_lat.sum(axis=(0, 1))
    k_lat = int(np.argmax(depth_sums_lat)) if depth_sums_lat.max() > 0 else ch0_lat.shape[2] // 2

    # Anatomy slice window
    anat_sl = image[:, :, k_img]
    v0 = float(anat_sl.min())
    v1 = float(anat_sl.max())
    if v1 <= v0:
        v0, v1 = 0.0, 1.0

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Mask QC — {patient_id} (native frame)", color="white", fontsize=11)

    col_labels = [
        [f"{roi_label} (hard)", "NETC (hard)"],
        [f"{roi_label} (soft, image res)", "NETC (soft, image res)"],
        [f"{roi_label} (latent grid, k={k_lat})", "NETC (latent grid)"],
    ]

    # Row 0: anatomy + hard mask (binary; high-contrast colours)
    for col in range(2):
        ax = axes[0, col]
        ax.set_facecolor("black")
        ax.imshow(np.rot90(anat_sl), cmap="gray", vmin=v0, vmax=v1)
        if hard_mask.ndim == 3:
            if col == 0:
                if roi_label.upper() == "TC":
                    hm_bin = ((hard_mask[:, :, k_img] > 0) & (hard_mask[:, :, k_img] != 2)).astype(
                        np.float32
                    )
                else:
                    hm_bin = (hard_mask[:, :, k_img] > 0).astype(np.float32)
            else:
                hm_bin = (hard_mask[:, :, k_img] == 1).astype(np.float32)
        else:
            ch = min(col, hard_mask.shape[0] - 1)
            hm_bin = (hard_mask[ch, :, :, k_img] > 0).astype(np.float32)
        arr_rot = np.rot90(hm_bin)
        ax.imshow(_overlay_rgba(arr_rot, _COLORS[col], alpha_max=0.6))
        ax.set_title(col_labels[0][col], color="white", fontsize=8)
        ax.axis("off")

    # Row 1: anatomy + soft mask (continuous probability colormap + contours)
    for col in range(2):
        ax = axes[1, col]
        ax.set_facecolor("black")
        ax.imshow(np.rot90(anat_sl), cmap="gray", vmin=v0, vmax=v1)
        soft_sl = soft_mask_img[col, :, :, k_img]
        arr_rot = np.rot90(soft_sl)
        ax.imshow(_overlay_cmap_rgba(arr_rot, _CMAPS[col], alpha_max=0.75))
        _add_contours(ax, arr_rot, _COLORS[col])
        ax.set_title(col_labels[1][col], color="white", fontsize=8)
        ax.axis("off")

    # Row 2: soft mask on the latent grid (black bg + continuous colour + contours)
    for col in range(2):
        ax = axes[2, col]
        ax.set_facecolor("black")
        lat_sl = soft_mask_latent[col, :, :, k_lat]
        arr_rot = np.rot90(lat_sl)
        ax.imshow(_overlay_cmap_rgba(arr_rot, _CMAPS[col], alpha_max=1.0))
        _add_contours(ax, arr_rot, _COLORS[col])
        ax.set_title(col_labels[2][col], color="white", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug("render_mask_qc (native) -> %s", path)
    return path


def _render_mask_qc_crop_frame(
    *,
    image: np.ndarray,
    hard_mask: np.ndarray,
    soft_mask_img: np.ndarray,
    soft_mask_latent: np.ndarray,
    patient_id: str,
    path: Path,
    roi_label: str,
    crop_spec: CropPadSpec,
) -> Path:
    """Crop-frame rendering backend for :func:`render_mask_qc`.

    All arrays are projected into the ``(192, 224, 192)`` crop frame and a
    single reference depth slice ``k`` is selected from the hard-TC area,
    ensuring all rows show the same physical z-level at the same resolution.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    from vena.common import apply_crop_pad

    # --- Crop image (H,W,D) → (192,224,192) --------------------------------
    img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    img_crop: np.ndarray = apply_crop_pad(img_t, crop_spec).squeeze().numpy()

    # --- Crop hard_mask → (192,224,192) -------------------------------------
    if hard_mask.ndim == 3:
        hm_t = torch.from_numpy(hard_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        hard_crop: np.ndarray = (
            apply_crop_pad(hm_t, crop_spec).squeeze().numpy().round().astype(np.int32)
        )
    else:
        # (2,H,W,D) pre-binarized
        hm_t = torch.from_numpy(hard_mask.astype(np.float32)).unsqueeze(0)
        hard_crop = apply_crop_pad(hm_t, crop_spec).squeeze(0).numpy()

    # --- Crop soft_mask_img (2,H,W,D) → (2,192,224,192) --------------------
    sm_t = torch.from_numpy(soft_mask_img).unsqueeze(0)  # (1,2,H,W,D)
    soft_img_crop: np.ndarray = apply_crop_pad(sm_t, crop_spec).squeeze(0).numpy()

    # --- Upscale soft_mask_latent (2,48,56,48) → (2,192,224,192) -----------
    lat_t = torch.from_numpy(soft_mask_latent).float().unsqueeze(0)
    lat_up: np.ndarray = (
        F.interpolate(lat_t, scale_factor=4, mode="trilinear", align_corners=False)
        .squeeze(0)
        .numpy()
    )

    # --- Reference slice: argmax of hard TC area in crop frame --------------
    if hard_crop.ndim == 3:
        if roi_label.upper() == "TC":
            tc_hard = ((hard_crop > 0) & (hard_crop != 2)).astype(np.float32)
        else:
            tc_hard = (hard_crop > 0).astype(np.float32)
    else:
        tc_hard = (hard_crop[0] > 0).astype(np.float32)

    depth_sums = tc_hard.sum(axis=(0, 1))  # (192,) depth profile
    k = int(np.argmax(depth_sums)) if depth_sums.max() > 0 else img_crop.shape[2] // 2

    # --- Anatomy slice (same k for all rows) --------------------------------
    anat_sl = img_crop[:, :, k]
    v0 = float(anat_sl.min())
    v1 = float(anat_sl.max())
    if v1 <= v0:
        v0, v1 = 0.0, 1.0

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Mask QC — {patient_id} (crop frame, k={k})", color="white", fontsize=11)

    col_labels = [
        [f"{roi_label} (hard)", "NETC (hard)"],
        [f"{roi_label} (soft, image)", "NETC (soft, image)"],
        [f"{roi_label} (latent ×4)", "NETC (latent ×4)"],
    ]

    # Row 0: anatomy + hard mask (binary overlay; existing green/magenta)
    for col in range(2):
        ax = axes[0, col]
        ax.set_facecolor("black")
        ax.imshow(np.rot90(anat_sl), cmap="gray", vmin=v0, vmax=v1)
        hm_bin = _hard_slice(hard_crop, k, col, roi_label)
        ax.imshow(_overlay_rgba(np.rot90(hm_bin), _COLORS[col], alpha_max=0.6))
        ax.set_title(col_labels[0][col], color="white", fontsize=8)
        ax.axis("off")

    # Row 1: anatomy + true sigmoid soft (perceptual colormap + contours)
    for col in range(2):
        ax = axes[1, col]
        ax.set_facecolor("black")
        ax.imshow(np.rot90(anat_sl), cmap="gray", vmin=v0, vmax=v1)
        soft_sl = soft_img_crop[col, :, :, k]
        arr_rot = np.rot90(soft_sl)
        ax.imshow(_overlay_cmap_rgba(arr_rot, _CMAPS[col], alpha_max=0.75))
        _add_contours(ax, arr_rot, _COLORS[col])
        ax.set_title(col_labels[1][col], color="white", fontsize=8)
        ax.axis("off")

    # Row 2: anatomy + latent-upscaled soft (perceptual colormap + contours)
    # Anatomy shown underneath so spatial alignment against the real MRI is
    # immediately visible — all three rows are now directly comparable.
    for col in range(2):
        ax = axes[2, col]
        ax.set_facecolor("black")
        ax.imshow(np.rot90(anat_sl), cmap="gray", vmin=v0, vmax=v1)
        lat_sl = lat_up[col, :, :, k]
        arr_rot = np.rot90(lat_sl)
        ax.imshow(_overlay_cmap_rgba(arr_rot, _CMAPS[col], alpha_max=0.80))
        _add_contours(ax, arr_rot, _COLORS[col])
        ax.set_title(col_labels[2][col], color="white", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug("render_mask_qc (crop-frame k=%d) -> %s", k, path)
    return path


def _hard_slice(
    hard_crop: np.ndarray,
    k: int,
    col: int,
    roi_label: str,
) -> np.ndarray:
    """Extract a binary hard-mask slice from *hard_crop* at depth *k*.

    Parameters
    ----------
    hard_crop : np.ndarray
        Either ``(H, W, D)`` integer label or ``(2, H, W, D)`` binary.
    k : int
        Depth index.
    col : int
        0 = channel-0 (TC/WT), 1 = NETC.
    roi_label : str
        ``"TC"`` triggers edema exclusion; any other value uses ``label > 0``.

    Returns
    -------
    np.ndarray
        2-D float32 binary mask ``(H, W)``.
    """
    if hard_crop.ndim == 3:
        sl = hard_crop[:, :, k]
        if col == 0:
            if roi_label.upper() == "TC":
                return ((sl > 0) & (sl != 2)).astype(np.float32)
            return (sl > 0).astype(np.float32)
        return (sl == 1).astype(np.float32)
    # (2, H, W, D) pre-binarized
    ch = min(col, hard_crop.shape[0] - 1)
    return (hard_crop[ch, :, :, k] > 0).astype(np.float32)


def render_slice_montage(
    patients: list[PatientView],
    *,
    n_cols: int = 10,
    alpha: float = 0.6,
    path: Path,
) -> Path:
    """Produce a multi-patient montage with the pinned layout.

    One patient per row, ordered by **ascending** tumour volume (small →
    large).  Each row contains exactly *n_cols* tumour-bearing axial slices
    (evenly spaced through the tumour extent); each cell shows a T1pre slice
    with the soft ``[WT, NETC]`` mask overlaid at *alpha*.

    Parameters
    ----------
    patients : list[PatientView]
        Patient data bundles.  Sorted by ``tumor_volume`` inside.
    n_cols : int
        Number of tumour-bearing slice columns per row.  Default 10.
    alpha : float
        Overlay opacity.  Default 0.6.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        *path* after writing.

    Raises
    ------
    SegMetricError
        If *patients* is empty.
    """
    if not patients:
        raise SegMetricError("patients list is empty; cannot build montage")

    ordered = sorted(patients, key=lambda p: p.tumor_volume)
    n_rows = len(ordered)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.patch.set_facecolor("black")

    # Normalise axes to always be 2-D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for r, pv in enumerate(ordered):
        z_indices = _axial_tumor_slices(pv.soft_mask, n_cols=n_cols)
        for c_idx, k in enumerate(z_indices):
            k = int(k)
            ax = axes[r, c_idx]
            ax.set_facecolor("black")

            anat_sl = pv.t1pre[:, :, k]
            v0 = float(anat_sl.min())
            v1 = float(anat_sl.max())
            if v1 <= v0:
                v0, v1 = 0.0, 1.0
            ax.imshow(np.rot90(anat_sl), cmap="gray", vmin=v0, vmax=v1)

            # WT overlay (continuous probability, green + contours)
            wt_sl = pv.soft_mask[0, :, :, k]
            wt_rot = np.rot90(wt_sl)
            ax.imshow(_overlay_rgba(wt_rot, _WT_COLOR, alpha_max=alpha))
            _add_contours(ax, wt_rot, _WT_COLOR)

            # NETC overlay (continuous probability, magenta + contours)
            netc_sl = pv.soft_mask[1, :, :, k]
            netc_rot = np.rot90(netc_sl)
            ax.imshow(_overlay_rgba(netc_rot, _NETC_COLOR, alpha_max=alpha))
            _add_contours(ax, netc_rot, _NETC_COLOR)

            ax.axis("off")
            if c_idx == 0:
                ax.set_title(
                    f"{pv.patient_id}\nvol={pv.tumor_volume:.0f}v",
                    color="white",
                    fontsize=7,
                    loc="left",
                )

    fig.tight_layout(pad=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug("render_slice_montage -> %s  rows=%d cols=%d", path, n_rows, n_cols)
    return path


def render_latent_embedding(
    mask_latents: dict[str, Any],
    meta: pd.DataFrame,
    *,
    method: str = "pca_umap_perpatient",
    color_by: tuple[str, ...] = ("tumor_volume", "cohort"),
    path: Path,
) -> Path:
    """Produce a 2-D per-patient embedding of flattened mask-latent vectors.

    PCA is used as the primary embedding method.  UMAP is tried when
    ``method`` contains ``"umap"`` and ``umap-learn`` is importable;
    otherwise falls back to PCA with a logged warning.

    Parameters
    ----------
    mask_latents : dict[str, array-like]
        Maps patient ID → mask-latent array, shape ``(2, *LATENT_SPATIAL)``.
    meta : pd.DataFrame
        Must be indexed by patient ID (or have a ``"patient_id"`` column)
        and contain at least the columns listed in *color_by*.
    method : str
        Embedding method key (``"pca_umap_perpatient"`` = try UMAP, fall
        back to PCA).
    color_by : tuple[str, ...]
        Column names in *meta* to use for colour-coding.  One sub-plot
        per entry.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        *path* after writing.

    Raises
    ------
    SegMetricError
        If *mask_latents* is empty.
    """
    if not mask_latents:
        raise SegMetricError("mask_latents is empty; nothing to embed")

    # Build ordered patient list + feature matrix
    pids = sorted(mask_latents.keys())
    feat_mat = np.stack([_to_numpy(mask_latents[pid]).ravel() for pid in pids])  # (N, D)

    # Align meta to the patient list
    if "patient_id" in meta.columns:
        meta_indexed = meta.set_index("patient_id")
    else:
        meta_indexed = meta

    # Choose embedding method
    use_umap = "umap" in method and importlib.util.find_spec("umap") is not None
    if "umap" in method and not use_umap:
        logger.warning(
            "umap-learn is not installed; falling back to PCA for latent embedding. "
            "Install umap-learn to use UMAP."
        )

    if use_umap:
        import umap  # type: ignore[import]

        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(feat_mat)
        embed_label = "UMAP"
    else:
        from sklearn.decomposition import PCA

        n_comp = min(2, feat_mat.shape[0], feat_mat.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        embedding_raw = pca.fit_transform(feat_mat)
        # Pad to 2 columns if only 1 patient
        if embedding_raw.shape[1] < 2:
            embedding = np.hstack(
                [embedding_raw, np.zeros((embedding_raw.shape[0], 2 - embedding_raw.shape[1]))]
            )
        else:
            embedding = embedding_raw
        embed_label = "PCA"

    n_panels = len(color_by)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.patch.set_facecolor("black")
    if n_panels == 1:
        axes = [axes]

    for ax, col_key in zip(axes, color_by, strict=False):
        ax.set_facecolor("black")
        try:
            col_vals = meta_indexed.loc[pids, col_key].values
        except KeyError:
            logger.warning("metadata column %r not found; skipping colour", col_key)
            col_vals = np.zeros(len(pids))

        if np.issubdtype(np.array(col_vals).dtype, np.number):
            sc = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=col_vals.astype(float),
                cmap="viridis",
                s=40,
                alpha=0.9,
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.ax.yaxis.label.set_color("white")
            cbar.ax.tick_params(colors="white")
        else:
            # Categorical colour coding
            categories = list(dict.fromkeys(col_vals))  # preserve order, deduplicate
            cat_to_idx = {c: i for i, c in enumerate(categories)}
            c_idx = np.array([cat_to_idx[v] for v in col_vals])
            cmap_cat = plt.get_cmap("tab10")
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=cmap_cat(c_idx % 10),
                s=40,
                alpha=0.9,
            )
            for cat in categories:
                ax.scatter([], [], c=[cmap_cat(cat_to_idx[cat] % 10)], label=str(cat))
            ax.legend(
                facecolor="#222222",
                labelcolor="white",
                fontsize=7,
                loc="best",
            )

        ax.set_title(f"{embed_label} — colour: {col_key}", color="white", fontsize=9)
        ax.set_xlabel(f"{embed_label}-1", color="white", fontsize=8)
        ax.set_ylabel(f"{embed_label}-2", color="white", fontsize=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    fig.suptitle("Mask-latent embedding (per-patient)", color="white", fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug("render_latent_embedding -> %s  method=%s", path, embed_label)
    return path


def render_injection_sanity(
    module: Any,
    batch: dict[str, Any],
    *,
    path: Path,
) -> Path:
    """Visualise ControlNet residual locality at step 0 and at output_scale > 0.

    **S2 deliverable** — call with synthetic data in Phase-1 testing; do not
    run over a real FM module until injection is wired (Phase 2).

    Expected *batch* keys:

    ``wt_mask`` : np.ndarray or Tensor
        WT mask at image (or latent) resolution, shape ``(H, W, D)``
        or ``(2, H, W, D)`` (WT channel 0 used).
    ``residuals_zero`` : np.ndarray or Tensor
        Per-voxel residual map at ``output_scale = 0``, shape
        ``(H, W, D)`` or ``(C, H, W, D)``.  Should be ≈ 0 everywhere.
    ``residuals_scale`` : np.ndarray or Tensor
        Per-voxel residual map at ``output_scale > 0``, shape
        ``(H, W, D)`` or ``(C, H, W, D)``.  Should be concentrated
        inside the WT region.

    *module* is accepted for API compatibility with Phase 2; it is not
    called in this Phase-1 implementation.

    Parameters
    ----------
    module : Any
        FM LightningModule (ignored in Phase 1; pass ``None`` for tests).
    batch : dict[str, Any]
        See above.
    path : Path
        Output PNG path.

    Returns
    -------
    Path
        *path* after writing.

    Raises
    ------
    SegMetricError
        If *batch* is missing required keys.
    """
    required = {"wt_mask", "residuals_zero", "residuals_scale"}
    missing = required - set(batch.keys())
    if missing:
        raise SegMetricError(f"batch is missing required keys: {sorted(missing)}")

    wt_mask = _to_numpy(batch["wt_mask"]).astype(np.float32)
    res_zero = _to_numpy(batch["residuals_zero"]).astype(np.float32)
    res_scale = _to_numpy(batch["residuals_scale"]).astype(np.float32)

    # Collapse channel dims if present
    if res_zero.ndim == 4:
        res_zero = np.sqrt((res_zero**2).sum(axis=0))
    if res_scale.ndim == 4:
        res_scale = np.sqrt((res_scale**2).sum(axis=0))
    if wt_mask.ndim == 4:
        wt_mask = wt_mask[0]

    ratio = compute_residual_energy_ratio(res_scale, wt_mask)

    # Pick the depth slice with the largest WT area (sum over in-plane axes)
    depth_sums = wt_mask.sum(axis=(0, 1)) if wt_mask.ndim == 3 else wt_mask
    k = int(np.argmax(depth_sums)) if depth_sums.max() > 0 else wt_mask.shape[2] // 2

    def _sl(vol3d: np.ndarray) -> np.ndarray:
        return vol3d[:, :, k]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("black")

    panel_data = [
        (np.rot90(_sl(wt_mask)), "WT mask (reference)", "hot"),
        (np.rot90(_sl(res_zero)), "Residual @ scale=0  (should be ≈ 0)", "inferno"),
        (np.rot90(_sl(res_scale)), f"Residual @ scale>0  (in/out ratio={ratio:.2f})", "inferno"),
    ]

    for ax, (data, title, cmap) in zip(axes, panel_data, strict=False):
        ax.set_facecolor("black")
        v_max = max(float(data.max()), 1e-8)
        ax.imshow(data, cmap=cmap, vmin=0.0, vmax=v_max)
        ax.set_title(title, color="white", fontsize=8, wrap=True)
        ax.axis("off")

    fig.suptitle(
        f"Injection sanity — in-WT/out-WT energy ratio = {ratio:.3f}",
        color="white",
        fontsize=10,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug(
        "render_injection_sanity -> %s  in/out_ratio=%.3f  res_zero_max=%.2e",
        path,
        ratio,
        float(res_zero.max()),
    )
    return path

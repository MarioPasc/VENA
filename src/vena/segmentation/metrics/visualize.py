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

__all__ = [
    "PatientView",
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
        Shape ``(2, H, W, D)``.  Channel 0 = WT.
    n_cols : int
        Number of slices to return.

    Returns
    -------
    np.ndarray
        1-D int array of length *n_cols*, depth indices into axis 2.
    """
    wt = soft_mask[0]  # (H, W, D)
    depth_presence = wt.max(axis=(0, 1)) > 0  # (D,)
    z_tumour = np.where(depth_presence)[0]
    if len(z_tumour) == 0:
        d = wt.shape[2]
        return np.linspace(0, d - 1, n_cols, dtype=int)
    return np.linspace(z_tumour[0], z_tumour[-1], n_cols, dtype=int)


def _overlay_rgba(
    soft_ch: np.ndarray,
    cmap_name: str,
    alpha: float,
) -> np.ndarray:
    """Build a (H, W, 4) RGBA overlay where the alpha channel = soft_ch * alpha.

    Parameters
    ----------
    soft_ch : np.ndarray
        2-D float array in ``[0, 1]``, shape ``(H, W)``.
    cmap_name : str
        Matplotlib colormap name.
    alpha : float
        Maximum overlay opacity (scales the per-pixel alpha).

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 4)`` float32 RGBA image.
    """
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(soft_ch).astype(np.float32)  # (H, W, 4)
    rgba[..., 3] = (soft_ch * alpha).astype(np.float32)
    return rgba


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
        Soft ``[WT, NETC]`` probability map at **image** resolution,
        shape ``(2, H, W, D)``, float in ``[0, 1]``.
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


# ---------------------------------------------------------------------------
# Machine-stats helpers
# ---------------------------------------------------------------------------


def compute_mask_stats(soft_masks: np.ndarray) -> dict[str, float | int]:
    """Compute machine stats from a batch of soft masks.

    Parameters
    ----------
    soft_masks : np.ndarray
        Float32 array of shape ``(N, 2, H, W, D)`` in ``[0, 1]``.
        Channel 0 = WT, channel 1 = NETC.

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
) -> Path:
    """Produce a 3-row QC figure for a single patient.

    Row 0: T1pre anatomy + hard-mask overlay (WT | NETC).
    Row 1: T1pre anatomy + soft-mask overlay at image resolution (WT | NETC).
    Row 2: soft-mask displayed on the ``(48, 56, 48)`` latent grid (WT | NETC).

    Parameters
    ----------
    image : np.ndarray
        T1pre volume, shape ``(H, W, D)``, float in ``[0, 1]``.
    hard_mask : np.ndarray
        Binary integer label map ``(H, W, D)`` (BraTS convention); or
        ``(2, H, W, D)`` pre-binarized per-channel.
    soft_mask_img : np.ndarray
        Soft ``[WT, NETC]`` map at image resolution, shape ``(2, H, W, D)``.
    soft_mask_latent : np.ndarray
        Soft ``[WT, NETC]`` map at latent grid, shape ``(2, *LATENT_SPATIAL)``.
    patient_id : str
        Label used in the figure suptitle.
    path : Path
        Output PNG path; parent directories must exist or will be created.

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

    # Pick the axial slice with maximum WT presence at image resolution
    wt_img = soft_mask_img[0]  # (H, W, D)
    depth_sums_img = wt_img.max(axis=(0, 1))
    k_img = int(np.argmax(depth_sums_img)) if depth_sums_img.max() > 0 else wt_img.shape[2] // 2

    # Pick the best latent-grid slice (depth axis = axis 2 of LATENT_SPATIAL)
    wt_lat = soft_mask_latent[0]  # (48, 56, 48)
    depth_sums_lat = wt_lat.max(axis=(0, 1))
    k_lat = int(np.argmax(depth_sums_lat)) if depth_sums_lat.max() > 0 else wt_lat.shape[2] // 2

    # Anatomy slice window
    anat_sl = image[:, :, k_img]
    v0 = float(anat_sl.min())
    v1 = float(anat_sl.max())
    if v1 <= v0:
        v0, v1 = 0.0, 1.0

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Mask QC — {patient_id}", color="white", fontsize=11)

    col_labels = [
        ["WT (hard)", "NETC (hard)"],
        ["WT (soft, image res)", "NETC (soft, image res)"],
        ["WT (latent grid)", "NETC (latent grid)"],
    ]
    cmap_names = ["hot", "cool"]

    # Row 0: anatomy + hard mask
    for col in range(2):
        ax = axes[0, col]
        ax.set_facecolor("black")
        ax.imshow(np.rot90(anat_sl), cmap="gray", vmin=v0, vmax=v1)
        if hard_mask.ndim == 3:
            # BraTS integer label: WT = label ∈ {1,2,3,4}; NETC = label == 1
            if col == 0:
                hm_bin = (hard_mask[:, :, k_img] > 0).astype(np.float32)
            else:
                hm_bin = (hard_mask[:, :, k_img] == 1).astype(np.float32)
        else:
            ch = min(col, hard_mask.shape[0] - 1)
            hm_bin = (hard_mask[ch, :, :, k_img] > 0).astype(np.float32)
        overlay = _overlay_rgba(np.rot90(hm_bin), cmap_names[col], alpha=0.6)
        ax.imshow(overlay)
        ax.set_title(col_labels[0][col], color="white", fontsize=8)
        ax.axis("off")

    # Row 1: anatomy + soft mask at image resolution
    for col in range(2):
        ax = axes[1, col]
        ax.set_facecolor("black")
        ax.imshow(np.rot90(anat_sl), cmap="gray", vmin=v0, vmax=v1)
        soft_sl = soft_mask_img[col, :, :, k_img]
        overlay = _overlay_rgba(np.rot90(soft_sl), cmap_names[col], alpha=0.7)
        ax.imshow(overlay)
        ax.set_title(col_labels[1][col], color="white", fontsize=8)
        ax.axis("off")

    # Row 2: soft mask on the latent grid
    for col in range(2):
        ax = axes[2, col]
        ax.set_facecolor("black")
        lat_sl = soft_mask_latent[col, :, :, k_lat]
        ax.imshow(np.rot90(lat_sl), cmap=cmap_names[col], vmin=0.0, vmax=1.0)
        ax.set_title(col_labels[2][col], color="white", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug("render_mask_qc -> %s", path)
    return path


def render_slice_montage(
    patients: list[PatientView],
    *,
    n_cols: int = 5,
    alpha: float = 0.7,
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
        Number of tumour-bearing slice columns per row.  Default 5.
    alpha : float
        Overlay opacity.  Default 0.7.
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

            # WT overlay (hot colormap)
            wt_sl = pv.soft_mask[0, :, :, k]
            ax.imshow(_overlay_rgba(np.rot90(wt_sl), "hot", alpha=alpha))

            # NETC overlay (cool colormap)
            netc_sl = pv.soft_mask[1, :, :, k]
            ax.imshow(_overlay_rgba(np.rot90(netc_sl), "cool", alpha=alpha))

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

    # Pick best depth slice in the wt_mask
    depth_sums = wt_mask.max(axis=(0, 1)) if wt_mask.ndim == 3 else wt_mask
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

"""QC figure generators for the MAISI encoding routine.

Three figure families:

* :func:`render_roundtrip_figure` — generic single figure: rows = supplied
  rows, columns = original / reconstructed / MAE / MSE / Lp³, all cells are
  three-plane composites anchored on the tumour centroid in image space.
* :func:`render_per_modality_roundtrip_figures` — wraps the above to emit
  one figure per modality, given a list of rows spanning modalities.
* :func:`render_pca_figure` — PCA of GAP-pooled latents.

All figures use a black background (mpl ``dark_background`` style) with
``magma`` error maps. The colour scheme is symmetric for the visible
slices (grayscale) and sequential for errors.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

logger = logging.getLogger(__name__)


_PLANE_LABELS = ("axial", "sagittal", "coronal")


@dataclass(frozen=True)
class RoundtripRow:
    """One row of the roundtrip figure: original + recon + 3 error maps per modality.

    Attributes
    ----------
    patient_id : str
    who_grade : int
    modality : str
    original : np.ndarray
        Shape ``(H, W, D)``, image space, percentile-normalised to ``[0, 1]``.
    reconstructed : np.ndarray
        Same shape, decoded image.
    tumor_centroid : tuple[int, int, int]
        ``(i, j, k)`` indices in image space.
    """

    patient_id: str
    who_grade: int
    modality: str
    original: np.ndarray
    reconstructed: np.ndarray
    tumor_centroid: tuple[int, int, int]


def _three_plane_slices(
    vol: np.ndarray,
    centroid: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract axial / sagittal / coronal slices at ``centroid``.

    Convention (matches UCSF-PDGM ``(H=240, W=240, D=155)``):
      * axial    = vol[:, :, k]
      * sagittal = vol[i, :, :]
      * coronal  = vol[:, j, :]

    All three are returned as 2-D arrays with the *first* axis vertical
    (i.e. as ``imshow`` expects).
    """
    i, j, k = centroid
    i = int(np.clip(i, 0, vol.shape[0] - 1))
    j = int(np.clip(j, 0, vol.shape[1] - 1))
    k = int(np.clip(k, 0, vol.shape[2] - 1))
    axial = vol[:, :, k].T[::-1]  # (W, H) so it reads anterior-up
    sagittal = vol[i, :, :].T[::-1]  # (D, W)
    coronal = vol[:, j, :].T[::-1]  # (D, H)
    return axial, sagittal, coronal


def _draw_three_plane_cell(
    gs: GridSpecFromSubplotSpec,
    fig: Figure,
    vol: np.ndarray,
    centroid: tuple[int, int, int],
    cmap: str,
    vmin: float,
    vmax: float,
    title: str | None,
) -> None:
    """Draw one composite cell: axial top, sagittal + coronal split below."""
    ax_axial = fig.add_subplot(gs[0, :])
    ax_sag = fig.add_subplot(gs[1, 0])
    ax_cor = fig.add_subplot(gs[1, 1])
    axial, sagittal, coronal = _three_plane_slices(vol, centroid)
    ax_axial.imshow(axial, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_sag.imshow(sagittal, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax_cor.imshow(coronal, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for ax in (ax_axial, ax_sag, ax_cor):
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        ax_axial.set_title(title, fontsize=8)


def render_roundtrip_figure(
    rows: list[RoundtripRow],
    output_path: Path,
    dpi: int = 200,
    title: str | None = "MAISI encode → decode roundtrip fidelity",
    row_label: Callable[[RoundtripRow], str] | None = None,
) -> Path:
    """Render the roundtrip figure (one row per (patient, modality) pair).

    Layout: ``n_rows × 5`` super-cells. Each super-cell is itself a 2×2
    sub-grid (axial spanning the top row, sagittal + coronal in the
    bottom row).

    Columns: ``[Original | Reconstructed | MAE | MSE | Lp³]``. The error
    maps use a perceptually-uniform sequential colormap (``magma``) with
    per-column vmax = the 99-th percentile across rows for that error
    type — comparable across patients within a column without saturating
    on a single outlier.

    The figure uses a black background (matplotlib's ``dark_background``
    style); pass a custom ``row_label`` callable to control the left-margin
    annotation when rendering multi-grade collages.
    """
    if not rows:
        raise ValueError("render_roundtrip_figure requires at least one row")

    error_maps_mae: list[np.ndarray] = []
    error_maps_mse: list[np.ndarray] = []
    error_maps_lp3: list[np.ndarray] = []
    for r in rows:
        diff = r.reconstructed - r.original
        error_maps_mae.append(np.abs(diff))
        error_maps_mse.append(diff**2)
        error_maps_lp3.append(np.abs(diff) ** 3)

    # Per-column normalisation.
    vmax_mae = max(_p99(e) for e in error_maps_mae)
    vmax_mse = max(_p99(e) for e in error_maps_mse)
    vmax_lp3 = max(_p99(e) for e in error_maps_lp3)

    if row_label is None:

        def row_label(r: RoundtripRow) -> str:
            return f"{r.patient_id}\nWHO {r.who_grade}\n{r.modality}"

    n_rows = len(rows)
    with plt.style.context("dark_background"):
        fig = plt.figure(
            figsize=(15, 3.5 * n_rows),
            facecolor="black",
        )
        outer = GridSpec(n_rows, 5, figure=fig, hspace=0.25, wspace=0.15)
        col_titles = ("Original", "Reconstructed", "MAE", "MSE", "Lp$^3$")
        for ri, r in enumerate(rows):
            cells = [
                (r.original, "gray", 0.0, 1.0),
                (r.reconstructed, "gray", 0.0, 1.0),
                (error_maps_mae[ri], "magma", 0.0, vmax_mae),
                (error_maps_mse[ri], "magma", 0.0, vmax_mse),
                (error_maps_lp3[ri], "magma", 0.0, vmax_lp3),
            ]
            for ci, (vol, cmap, vmin, vmax) in enumerate(cells):
                sub = GridSpecFromSubplotSpec(
                    2,
                    2,
                    subplot_spec=outer[ri, ci],
                    height_ratios=[2, 1],
                    hspace=0.02,
                    wspace=0.02,
                )
                title_str = col_titles[ci] if ri == 0 else None
                _draw_three_plane_cell(sub, fig, vol, r.tumor_centroid, cmap, vmin, vmax, title_str)
            fig.text(
                0.02,
                1.0 - (ri + 0.5) / n_rows,
                row_label(r),
                ha="left",
                va="center",
                fontsize=8,
                family="monospace",
                color="white",
            )

        if title is not None:
            fig.suptitle(title, fontsize=12, color="white")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="black")
        plt.close(fig)
    logger.info("Wrote roundtrip figure: %s (+ %s)", output_path, pdf_path)
    return output_path


def render_per_modality_roundtrip_figures(
    rows: list[RoundtripRow],
    output_dir: Path,
    filename_template: str = "roundtrip_{modality}.png",
    title_template: str | None = "MAISI roundtrip fidelity — {modality}",
    dpi: int = 200,
) -> dict[str, Path]:
    """Group ``rows`` by modality and emit one figure per modality.

    Returns a ``{modality: output_path}`` mapping.
    """
    by_modality: dict[str, list[RoundtripRow]] = {}
    for r in rows:
        by_modality.setdefault(r.modality, []).append(r)
    out: dict[str, Path] = {}
    for modality, modrows in by_modality.items():
        path = output_dir / filename_template.format(modality=modality)
        title = title_template.format(modality=modality) if title_template is not None else None
        render_roundtrip_figure(modrows, path, dpi=dpi, title=title)
        out[modality] = path
    return out


def _p99(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 1.0
    return float(np.percentile(arr, 99))


def render_pca_figure(
    pooled_latents: np.ndarray,
    modalities_per_row: list[str],
    tumor_volume_ml: np.ndarray,
    output_path: Path,
    n_components: int = 2,
    dpi: int = 200,
) -> Path:
    """PCA of GAP-pooled latents.

    Parameters
    ----------
    pooled_latents : np.ndarray
        Shape ``(R, C)`` with ``R = n_patients × n_modalities``. Each row
        is a globally-average-pooled latent vector.
    modalities_per_row : list[str]
        Length ``R``; the modality slug for each row.
    tumor_volume_ml : np.ndarray
        Length ``R``; the patient-level tumour volume (mL) replicated per
        modality row.
    output_path : Path
    n_components : int
        Number of PCs to fit; only the first two are plotted.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if pooled_latents.ndim != 2:
        raise ValueError(f"pooled_latents must be 2-D; got {pooled_latents.shape}")
    if len(modalities_per_row) != pooled_latents.shape[0]:
        raise ValueError("modalities_per_row length must match rows of pooled_latents")
    if tumor_volume_ml.shape[0] != pooled_latents.shape[0]:
        raise ValueError("tumor_volume_ml length must match rows of pooled_latents")

    scaler = StandardScaler().fit(pooled_latents)
    z = scaler.transform(pooled_latents)
    n_components = min(n_components, z.shape[1], z.shape[0])
    pca = PCA(n_components=n_components).fit(z)
    pcs = pca.transform(z)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor="black")
    markers = {"t1pre": "o", "t1c": "s", "t2": "^", "flair": "D", "adc": "P", "swi": "X"}
    # Hue: use viridis on the (clipped) tumour volume.
    vmax = (
        float(np.nanpercentile(tumor_volume_ml, 95)) if np.isfinite(tumor_volume_ml).any() else 1.0
    )
    vmax = max(vmax, 1e-3)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=vmax)

    for mod in sorted(set(modalities_per_row)):
        mask = np.asarray([m == mod for m in modalities_per_row])
        sc = ax.scatter(
            pcs[mask, 0],
            pcs[mask, 1] if pcs.shape[1] > 1 else np.zeros(mask.sum()),
            c=tumor_volume_ml[mask],
            cmap=cmap,
            norm=norm,
            marker=markers.get(mod, "o"),
            edgecolor="black",
            linewidth=0.3,
            s=40,
            alpha=0.85,
            label=mod,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    if pcs.shape[1] > 1:
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA of GAP-pooled MAISI latents")

    ax.legend(title="modality", loc="best", fontsize=8)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.7,
        label="tumour volume NETC+ET (mL)",
    )
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    plt.style.use("default")
    logger.info("Wrote PCA figure: %s (+ %s)", output_path, pdf_path)
    return output_path

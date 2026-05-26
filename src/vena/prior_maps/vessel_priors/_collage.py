"""Per-patient QC collage for vessel-prior outputs.

Layout (5 rows × 7 image columns, 3 + gap + 1 + gap + 3):

    +---+---+---+   +-----+   +---+---+---+
    |axi|sag|cor|   | mid |   |axi|sag|cor|
    +---+---+---+   +-----+   +---+---+---+
            ... 5 rows ...

* Left group   : grayscale SWI slices in axial / sagittal / coronal planes.
* Middle col   : axial slice of the soft Frangi response (``hot`` cmap, black
                 at 0) plus a contour of the thresholded binary mask in a
                 highlight colour (alpha = 1.0).
* Right group  : grayscale SWI in axi / sag / cor with the soft response
                 alpha-blended on top (alpha ∝ response, capped at 0.7) and
                 the binary-mask contour drawn at alpha = 1.0.

Rows correspond to 5 evenly-spaced slice positions selected from the *list of
non-empty indices per axis* (i.e. slices the brain mask actually touches), so
no row is blank by construction. Slice indices are chosen independently per
plane: axial Z's for the axial column, sagittal X's for the sagittal column,
coronal Y's for the coronal column. The middle column uses the axial Z's.

Background is solid black across the whole figure; the soft-response colormap
has black at value 0 so empty regions blend into the axes face.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from vena.data.niigz import non_empty_indices, pick_evenly_from

# Colour conventions kept central so re-tuning is one-line.
_SOFT_CMAP = "hot"  # black -> red -> yellow -> white; black at 0
_CONTOUR_COLOR = "#00ffff"  # bright cyan, complementary to the hot colormap
_BACKGROUND = "black"


def _normalize_for_display(
    img: NDArray[Any], lo_pct: float = 0.5, hi_pct: float = 99.5
) -> NDArray[np.float32]:
    """Percentile-stretch a 2D slice to ``[0, 1]`` for grayscale display."""
    if img.size == 0:
        return img.astype(np.float32)
    lo = float(np.percentile(img, lo_pct))
    hi = float(np.percentile(img, hi_pct))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def _take_slice(vol: NDArray[Any], idx: int, axis: int) -> NDArray[Any]:
    """``np.take``-style slice, then rotate 90° CCW for display."""
    s = np.take(vol, idx, axis=axis)
    return np.rot90(s, k=1)


def _style_panel(ax: Axes) -> None:
    """Strip ticks / spines and set black face for a single panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(_BACKGROUND)


def _draw_swi(ax: Axes, slc: NDArray[Any]) -> None:
    ax.imshow(
        _normalize_for_display(slc),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )


def _draw_soft_with_contour(ax: Axes, soft_slc: NDArray[Any], binary_slc: NDArray[Any]) -> None:
    """Middle-column rendering: soft response (black-base) + binary contour."""
    ax.imshow(
        soft_slc,
        cmap=_SOFT_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    if np.any(binary_slc):
        ax.contour(
            binary_slc.astype(np.float32),
            levels=[0.5],
            colors=[_CONTOUR_COLOR],
            linewidths=0.8,
            alpha=1.0,
        )


def _draw_overlay(
    ax: Axes,
    swi_slc: NDArray[Any],
    soft_slc: NDArray[Any],
    binary_slc: NDArray[Any],
    *,
    overlay_alpha: float,
) -> None:
    """Right-group rendering: SWI grayscale + soft overlay + binary contour."""
    ax.imshow(
        _normalize_for_display(swi_slc),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.imshow(
        soft_slc,
        cmap=_SOFT_CMAP,
        vmin=0.0,
        vmax=1.0,
        alpha=np.clip(soft_slc * overlay_alpha, 0.0, overlay_alpha),
        interpolation="nearest",
    )
    if np.any(binary_slc):
        ax.contour(
            binary_slc.astype(np.float32),
            levels=[0.5],
            colors=[_CONTOUR_COLOR],
            linewidths=0.8,
            alpha=1.0,
        )


def render_collage(
    swi: NDArray[Any],
    brain: NDArray[Any],
    soft: NDArray[np.float32],
    binary: NDArray[np.uint8],
    out_path: Path,
    *,
    patient_id: str,
    n_slices: int = 5,
    dpi: int = 150,
    overlay_alpha: float = 0.7,
    min_voxels_per_slice: int = 2000,
) -> Path:
    """Render and save a per-patient 3 + 1 + 3 collage.

    Parameters
    ----------
    swi
        SWI / SWAN volume, used as the grayscale background.
    brain
        Binary or near-binary brain mask; drives slice selection per axis.
    soft
        Float32 vesselness response in ``[0, 1]``, same shape as ``swi``.
    binary
        Uint8 thresholded mask, same shape as ``swi``. Drives the contour.
    out_path
        PNG destination; parent directories are created on demand.
    patient_id
        Shown in the figure suptitle.
    n_slices
        Number of rows (slice positions). Default 5.
    dpi
        Output resolution.
    overlay_alpha
        Maximum alpha applied to the soft overlay in the right-group columns.

    Returns
    -------
    Path
        Resolved output path.
    """
    # Pick non-empty slice indices independently per anatomical axis. LPS:
    # X=axis 0, Y=axis 1, Z=axis 2. Sagittal=fixed X, coronal=fixed Y, axial=fixed Z.
    # `min_voxels_per_slice` rejects sliver slices at the brain boundary that
    # would render as near-empty panels.
    z_pool = _slice_pool(brain, axis=2, min_voxels=min_voxels_per_slice)
    y_pool = _slice_pool(brain, axis=1, min_voxels=min_voxels_per_slice)
    x_pool = _slice_pool(brain, axis=0, min_voxels=min_voxels_per_slice)
    z_idx = _pad_to(pick_evenly_from(z_pool, n_slices), n_slices)
    y_idx = _pad_to(pick_evenly_from(y_pool, n_slices), n_slices)
    x_idx = _pad_to(pick_evenly_from(x_pool, n_slices), n_slices)

    n_image_cols = 3 + 1 + 3
    fig = plt.figure(
        figsize=(2.0 * n_image_cols + 1.0, 2.0 * n_slices + 0.6),
        facecolor=_BACKGROUND,
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        nrows=n_slices,
        ncols=9,  # 3 image + 1 gap + 1 image + 1 gap + 3 image
        width_ratios=[1.0, 1.0, 1.0, 0.35, 1.0, 0.35, 1.0, 1.0, 1.0],
        wspace=0.04,
        hspace=0.06,
    )
    image_col_map = [0, 1, 2, 4, 6, 7, 8]  # skip gap columns 3 and 5

    column_titles = (
        "Axial",
        "Sagittal",
        "Coronal",
        "Soft + threshold contour",
        "Axial (overlay)",
        "Sagittal (overlay)",
        "Coronal (overlay)",
    )

    for r in range(n_slices):
        zr, yr, xr = z_idx[r], y_idx[r], x_idx[r]

        swi_axi = _take_slice(swi, zr, axis=2)
        swi_sag = _take_slice(swi, xr, axis=0)
        swi_cor = _take_slice(swi, yr, axis=1)

        soft_axi = _take_slice(soft, zr, axis=2)
        soft_sag = _take_slice(soft, xr, axis=0)
        soft_cor = _take_slice(soft, yr, axis=1)

        bin_axi = _take_slice(binary, zr, axis=2)
        bin_sag = _take_slice(binary, xr, axis=0)
        bin_cor = _take_slice(binary, yr, axis=1)

        # Left group — plain SWI.
        for c, slc in zip(image_col_map[:3], (swi_axi, swi_sag, swi_cor), strict=True):
            ax = fig.add_subplot(gs[r, c])
            _draw_swi(ax, slc)
            _style_panel(ax)
            if r == 0:
                ax.set_title(
                    column_titles[image_col_map.index(c)],
                    fontsize=10,
                    color="white",
                )

        # Middle column — axial soft response with binary contour.
        ax_mid = fig.add_subplot(gs[r, image_col_map[3]])
        _draw_soft_with_contour(ax_mid, soft_axi, bin_axi)
        _style_panel(ax_mid)
        if r == 0:
            ax_mid.set_title(column_titles[3], fontsize=10, color="white")

        # Right group — SWI + soft overlay + binary contour.
        for c, swi_slc, soft_slc, bin_slc in zip(
            image_col_map[4:],
            (swi_axi, swi_sag, swi_cor),
            (soft_axi, soft_sag, soft_cor),
            (bin_axi, bin_sag, bin_cor),
            strict=True,
        ):
            ax = fig.add_subplot(gs[r, c])
            _draw_overlay(ax, swi_slc, soft_slc, bin_slc, overlay_alpha=overlay_alpha)
            _style_panel(ax)
            if r == 0:
                ax.set_title(
                    column_titles[image_col_map.index(c)],
                    fontsize=10,
                    color="white",
                )

        # Per-row label on the left edge: axial Z (the row's primary anchor).
        left_ax = fig.axes[-n_image_cols]
        left_ax.set_ylabel(
            f"z = {zr}\nx = {xr}\ny = {yr}",
            fontsize=9,
            color="white",
            rotation=0,
            labelpad=28,
            va="center",
        )

    fig.suptitle(patient_id, fontsize=13, color="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BACKGROUND)
    plt.close(fig)
    return out_path


def _slice_pool(mask: NDArray[Any], axis: int, min_voxels: int) -> list[int]:
    """Non-empty slice indices along ``axis`` with at least ``min_voxels`` set.

    Falls back to the lower threshold of 1 voxel if no slice meets the strict
    cut-off (e.g. a degenerate mask).
    """
    pool = non_empty_indices(mask, axis=axis, min_voxels=min_voxels)
    if not pool:
        pool = non_empty_indices(mask, axis=axis, min_voxels=1)
    return pool


def _pad_to(idx: list[int], n: int) -> list[int]:
    """Pad ``idx`` to length ``n`` by repeating its last (or 0) value."""
    if not idx:
        return [0] * n
    if len(idx) >= n:
        return idx[:n]
    return idx + [idx[-1]] * (n - len(idx))

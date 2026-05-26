"""Per-patient QC collage for perfusion-prior outputs.

Layout mirrors :mod:`vena.prior_maps.vessel_priors._collage` (5 rows × 7 image
columns, 3 + 1 + 3) so a reader of one collage immediately recognises the
others:

    +---+---+---+   +-----+   +---+---+---+
    |axi|sag|cor|   | mid |   |axi|sag|cor|
    +---+---+---+   +-----+   +---+---+---+
            ... 5 rows ...

* Left group   : grayscale ASL (CBF) slices in axial / sagittal / coronal.
* Middle col   : axial slice of the squashed ``cbf`` channel (``hot`` cmap,
                 range ``[-1, 1]``). No contour — perfusion has no natural
                 threshold (``binary is None``).
* Right group  : grayscale ASL with the ``cbf`` channel alpha-blended on top.

Rows correspond to 5 evenly-spaced slice positions selected from the *list of
non-empty indices per axis* of the brain mask, so no row is blank.
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

_CHANNEL_CMAP = "hot"
_CONTOUR_COLOR = "#00ffff"
_BACKGROUND = "black"


def _normalize_for_display(
    img: NDArray[Any], lo_pct: float = 0.5, hi_pct: float = 99.5
) -> NDArray[np.float32]:
    if img.size == 0:
        return img.astype(np.float32)
    lo = float(np.percentile(img, lo_pct))
    hi = float(np.percentile(img, hi_pct))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def _take_slice(vol: NDArray[Any], idx: int, axis: int) -> NDArray[Any]:
    s = np.take(vol, idx, axis=axis)
    return np.rot90(s, k=1)


def _style_panel(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(_BACKGROUND)


def _draw_source(ax: Axes, slc: NDArray[Any]) -> None:
    ax.imshow(
        _normalize_for_display(slc),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )


def _draw_channel(
    ax: Axes,
    channel_slc: NDArray[Any],
    binary_slc: NDArray[Any] | None,
    *,
    vmin: float,
    vmax: float,
) -> None:
    ax.imshow(
        channel_slc,
        cmap=_CHANNEL_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if binary_slc is not None and np.any(binary_slc):
        ax.contour(
            binary_slc.astype(np.float32),
            levels=[0.5],
            colors=[_CONTOUR_COLOR],
            linewidths=0.8,
            alpha=1.0,
        )


def _draw_overlay(
    ax: Axes,
    source_slc: NDArray[Any],
    channel_slc: NDArray[Any],
    binary_slc: NDArray[Any] | None,
    *,
    vmin: float,
    vmax: float,
    overlay_alpha: float,
) -> None:
    ax.imshow(
        _normalize_for_display(source_slc),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    span = max(vmax - vmin, 1e-6)
    alpha_map = np.clip((channel_slc - vmin) / span, 0.0, 1.0) * overlay_alpha
    ax.imshow(
        channel_slc,
        cmap=_CHANNEL_CMAP,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha_map,
        interpolation="nearest",
    )
    if binary_slc is not None and np.any(binary_slc):
        ax.contour(
            binary_slc.astype(np.float32),
            levels=[0.5],
            colors=[_CONTOUR_COLOR],
            linewidths=0.8,
            alpha=1.0,
        )


def render_collage(
    source: NDArray[Any],
    brain: NDArray[Any],
    channel: NDArray[np.float32],
    binary: NDArray[np.uint8] | None,
    out_path: Path,
    *,
    patient_id: str,
    source_label: str = "ASL (CBF)",
    channel_label: str = "cbf (tanh-squashed)",
    channel_vmin: float = -1.0,
    channel_vmax: float = 1.0,
    n_slices: int = 5,
    dpi: int = 150,
    overlay_alpha: float = 0.7,
    min_voxels_per_slice: int = 2000,
) -> Path:
    """Render and save a per-patient 3 + 1 + 3 collage for a perfusion channel.

    Parameters
    ----------
    source
        ASL CBF volume (used as the grayscale background).
    brain
        Binary brain mask; drives slice selection per axis.
    channel
        Float32 conditioning channel in ``[channel_vmin, channel_vmax]``.
    binary
        Optional uint8 contour mask, same shape as ``source``. ``None`` skips
        contour rendering — perfusion has no natural binary mask.
    out_path
        PNG destination; parent directories are created on demand.
    patient_id
        Shown in the figure suptitle.
    source_label, channel_label
        Column-title prefixes.
    channel_vmin, channel_vmax
        Colour-scale bounds for the channel column.
    """
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
        ncols=9,
        width_ratios=[1.0, 1.0, 1.0, 0.35, 1.0, 0.35, 1.0, 1.0, 1.0],
        wspace=0.04,
        hspace=0.06,
    )
    image_col_map = [0, 1, 2, 4, 6, 7, 8]

    column_titles = (
        f"{source_label} — Axial",
        f"{source_label} — Sagittal",
        f"{source_label} — Coronal",
        f"{channel_label} (axial)",
        f"{source_label} + {channel_label} — Axial",
        f"{source_label} + {channel_label} — Sagittal",
        f"{source_label} + {channel_label} — Coronal",
    )

    for r in range(n_slices):
        zr, yr, xr = z_idx[r], y_idx[r], x_idx[r]

        src_axi = _take_slice(source, zr, axis=2)
        src_sag = _take_slice(source, xr, axis=0)
        src_cor = _take_slice(source, yr, axis=1)

        ch_axi = _take_slice(channel, zr, axis=2)
        ch_sag = _take_slice(channel, xr, axis=0)
        ch_cor = _take_slice(channel, yr, axis=1)

        if binary is not None:
            bin_axi = _take_slice(binary, zr, axis=2)
            bin_sag = _take_slice(binary, xr, axis=0)
            bin_cor = _take_slice(binary, yr, axis=1)
        else:
            bin_axi = bin_sag = bin_cor = None

        for c, slc in zip(image_col_map[:3], (src_axi, src_sag, src_cor), strict=True):
            ax = fig.add_subplot(gs[r, c])
            _draw_source(ax, slc)
            _style_panel(ax)
            if r == 0:
                ax.set_title(
                    column_titles[image_col_map.index(c)],
                    fontsize=9,
                    color="white",
                )

        ax_mid = fig.add_subplot(gs[r, image_col_map[3]])
        _draw_channel(
            ax_mid,
            ch_axi,
            bin_axi,
            vmin=channel_vmin,
            vmax=channel_vmax,
        )
        _style_panel(ax_mid)
        if r == 0:
            ax_mid.set_title(column_titles[3], fontsize=9, color="white")

        for c, src_slc, ch_slc, bin_slc in zip(
            image_col_map[4:],
            (src_axi, src_sag, src_cor),
            (ch_axi, ch_sag, ch_cor),
            (bin_axi, bin_sag, bin_cor),
            strict=True,
        ):
            ax = fig.add_subplot(gs[r, c])
            _draw_overlay(
                ax,
                src_slc,
                ch_slc,
                bin_slc,
                vmin=channel_vmin,
                vmax=channel_vmax,
                overlay_alpha=overlay_alpha,
            )
            _style_panel(ax)
            if r == 0:
                ax.set_title(
                    column_titles[image_col_map.index(c)],
                    fontsize=9,
                    color="white",
                )

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
    pool = non_empty_indices(mask, axis=axis, min_voxels=min_voxels)
    if not pool:
        pool = non_empty_indices(mask, axis=axis, min_voxels=1)
    return pool


def _pad_to(idx: list[int], n: int) -> list[int]:
    if not idx:
        return [0] * n
    if len(idx) >= n:
        return idx[:n]
    return idx + [idx[-1]] * (n - len(idx))

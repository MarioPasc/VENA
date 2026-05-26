"""Mosaic + scatter plot helpers, returned as in-memory PNG bytes."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def _normalize(img: NDArray, lo: float = 0.5, hi: float = 99.5) -> NDArray:
    if img.size == 0:
        return img.astype(np.float32)
    p_lo = float(np.percentile(img, lo))
    p_hi = float(np.percentile(img, hi))
    if p_hi <= p_lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0).astype(np.float32)


def mosaic_three_axis(
    volumes: dict[str, NDArray],
    brain_mask: NDArray,
    out_path: Path,
    *,
    dpi: int = 100,
) -> Path:
    """Render a (n_volumes × 3) mosaic — axial, sagittal, coronal mid-slices.

    The grayscale volumes use percentile normalisation; the soft / channel
    volumes use the ``hot`` cmap in [-1, 1] or [0, 1] depending on min/max.
    """
    items = list(volumes.items())
    n = len(items)
    if n == 0:
        return out_path
    brain_bool = np.asarray(brain_mask) > 0
    # Mid-slices anchored on the brain centre of mass
    coords = np.array(np.where(brain_bool))
    if coords.size == 0:
        zc, yc, xc = (s // 2 for s in next(iter(volumes.values())).shape)
    else:
        xc, yc, zc = (int(c.mean()) for c in coords)

    fig, axes = plt.subplots(n, 3, figsize=(3 * 2.5, n * 2.5), facecolor="black", squeeze=False)
    for row, (name, vol) in enumerate(items):
        vol = np.asarray(vol)
        vmin, vmax = float(vol.min()), float(vol.max())
        is_gray = "T1" in name or "swan" in name.lower() or "adc" in name.lower()
        is_signed = vmin < -0.1
        cmap = "gray" if is_gray else "hot"
        norm_kw = dict(vmin=vmin, vmax=vmax)
        for col, (axis, idx) in enumerate(((2, zc), (0, xc), (1, yc))):
            ax = axes[row][col]
            slc = np.take(vol, idx, axis=axis)
            slc = np.rot90(slc, k=1)
            if is_gray:
                ax.imshow(_normalize(slc), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            else:
                if is_signed:
                    ax.imshow(slc, cmap="seismic", vmin=-1, vmax=1, interpolation="nearest")
                else:
                    ax.imshow(slc, cmap=cmap, **norm_kw, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_facecolor("black")
            if col == 0:
                ax.set_ylabel(
                    name,
                    color="white",
                    fontsize=9,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=30,
                )
        for col in range(3):
            if row == 0:
                axes[row][col].set_title(
                    ("Axial", "Sagittal", "Coronal")[col], color="white", fontsize=10
                )
    fig.suptitle("Visual QC strip", color="white", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return out_path


def scatter_to_png_bytes(x: NDArray, y: NDArray, *, title: str, max_points: int = 2000) -> bytes:
    """Render a scatter (sub-sampled to max_points) to PNG bytes for embedding."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size > max_points:
        idx = np.linspace(0, x.size - 1, max_points).astype(int)
        x = x[idx]
        y = y[idx]
    fig, ax = plt.subplots(figsize=(2.2, 1.8), dpi=120)
    if x.size > 0:
        ax.scatter(x, y, s=2, alpha=0.3, color="#1f77b4", edgecolors="none")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("prior", fontsize=6)
    ax.set_ylabel("ΔT1 (z)", fontsize=6)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

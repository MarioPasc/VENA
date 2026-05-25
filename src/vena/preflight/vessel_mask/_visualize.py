"""Figure rendering for the vessel-mask preflight.

Two figure types:

* :func:`render_threshold_curves` — 1×3 panel summarising binary fraction,
  connected-component count, and skeleton voxels as a function of threshold,
  with one curve per tag and a shaded anatomical-fraction band on the first
  panel.
* :func:`render_consensus_overlay` — per-patient axial-slice montage showing
  each method's binary mask plus the agreement / disagreement map at the
  recommended thresholds.

Both write to disk and return the saved path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from .analysis import PerTagSummary  # noqa: E402

logger = logging.getLogger(__name__)


def render_threshold_curves(
    summaries: list[PerTagSummary],
    *,
    target_fraction_range: tuple[float, float],
    out_path: Path,
    recommended_per_tag: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Three-panel curves: binary fraction, N components, skeleton voxels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_tag: dict[str, list[PerTagSummary]] = {}
    for s in summaries:
        by_tag.setdefault(s.tag, []).append(s)
    for k in by_tag:
        by_tag[k] = sorted(by_tag[k], key=lambda s: s.threshold)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, (tag, items) in enumerate(sorted(by_tag.items())):
        ts = np.asarray([s.threshold for s in items])
        bf = np.asarray([s.binary_fraction_mean for s in items])
        bf_err = np.asarray([s.binary_fraction_std for s in items])
        ncc = np.asarray([s.n_components_median for s in items])
        skl = np.asarray([s.skeleton_voxels_median for s in items])
        c = colours[i % len(colours)]
        axes[0].errorbar(ts, bf, yerr=bf_err, marker="o", color=c, label=tag, capsize=3)
        axes[1].plot(ts, ncc, marker="o", color=c, label=tag)
        axes[2].plot(ts, skl, marker="o", color=c, label=tag)
        if recommended_per_tag and tag in recommended_per_tag:
            t_rec = float(recommended_per_tag[tag]["threshold"])
            for ax in axes:
                ax.axvline(t_rec, color=c, linestyle="--", alpha=0.4)

    lo, hi = target_fraction_range
    axes[0].axhspan(lo, hi, color="green", alpha=0.10, label=f"anatomical {lo:.2f}–{hi:.2f}")
    axes[0].set_ylabel("binary fraction (brain-restricted)")
    axes[0].set_xlabel("threshold")
    axes[0].set_title("Binary fraction vs threshold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("median # connected components")
    axes[1].set_yscale("log")
    axes[1].set_title("CC count vs threshold")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].set_xlabel("threshold")
    axes[2].set_ylabel("median skeleton voxels")
    axes[2].set_title("Skeleton length vs threshold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_consensus_overlay(
    *,
    swi: NDArray[np.floating[Any]],
    brain: NDArray[Any],
    binaries: dict[str, NDArray[Any]],
    out_path: Path,
    patient_id: str,
    n_slices: int = 5,
) -> Path:
    """Per-patient axial-slice montage of method-A, method-B, agreement.

    Only the first two entries of ``binaries`` (sorted by tag) are used; this
    is by design — the consensus diagnostic is pairwise. A third tag would
    require a different visualisation. The convention is:

    * green  → only method A fires
    * red    → only method B fires
    * yellow → both methods fire (consensus)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tags = sorted(binaries.keys())
    if len(tags) < 2:
        raise ValueError("Need at least two methods to render a consensus overlay")
    tag_a, tag_b = tags[0], tags[1]
    a = binaries[tag_a].astype(bool, copy=False)
    b = binaries[tag_b].astype(bool, copy=False)
    brain_b = brain.astype(bool, copy=False)

    # Restrict overlays to brain; outside-brain voxels are masked out.
    a_only = a & ~b & brain_b
    b_only = b & ~a & brain_b
    both = a & b & brain_b

    z_brain = np.where(brain_b.any(axis=(0, 1)))[0]
    if z_brain.size == 0:
        z_idx = [swi.shape[2] // 2] * n_slices
    else:
        z_idx = np.linspace(z_brain.min(), z_brain.max(), n_slices, dtype=int).tolist()

    fig, axes = plt.subplots(1, n_slices, figsize=(3.0 * n_slices, 3.5))
    if n_slices == 1:
        axes = np.array([axes])

    swi_norm = swi.astype(np.float32)
    in_brain = swi_norm[brain_b]
    if in_brain.size:
        lo = float(np.percentile(in_brain, 0.5))
        hi = float(np.percentile(in_brain, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        swi_norm = np.clip((swi_norm - lo) / (hi - lo), 0.0, 1.0)

    for ax, z in zip(axes, z_idx, strict=False):
        bg = swi_norm[:, :, z].T
        ax.imshow(bg, cmap="gray", origin="lower", vmin=0, vmax=1)
        overlay = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.float32)
        overlay[a_only[:, :, z].T] = (0.1, 0.9, 0.1, 0.65)  # green
        overlay[b_only[:, :, z].T] = (0.9, 0.1, 0.1, 0.65)  # red
        overlay[both[:, :, z].T] = (1.0, 0.9, 0.1, 0.85)  # yellow
        ax.imshow(overlay, origin="lower")
        ax.set_title(f"z={int(z)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{patient_id}    green={tag_a} only, red={tag_b} only, yellow=consensus",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path

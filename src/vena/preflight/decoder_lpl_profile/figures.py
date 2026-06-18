"""Matplotlib renderers for the ``decoder_lpl_profile`` deliverables.

Every function takes a DataFrame or list-of-dicts and writes a figure
file (PNG). The figures are intentionally simple — the preflight is a
diagnostic, not a paper — and the implementations live here so the
aggregator can call them without touching plotting code itself.

The module sets a non-interactive backend at import time so the engine
runs on headless loginexa / picasso nodes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def magnitude_curve(
    rows: Iterable[dict],
    *,
    out_path: Path,
    berrada_inverse_upscale: bool = True,
) -> None:
    """X = block_idx, Y = ``E[||phi_l||]``, one band per cohort.

    Overlays Berrada's inverse-upscale prediction (``1 / scale``) so the
    sign-check in §4.7c is visually obvious.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    by_cohort_block: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        by_cohort_block.setdefault(r["cohort"], {}).setdefault(int(r["block_idx"]), []).append(
            float(r["mean_norm"])
        )
    for cohort, per_block in by_cohort_block.items():
        blks = sorted(per_block)
        means = [float(np.mean(per_block[b])) for b in blks]
        ax.plot(blks, means, "o-", label=cohort, lw=1.2)

    if berrada_inverse_upscale:
        # Block 0–2 = scale 1; 3 = scale 2 boundary; 4–5 = scale 2; etc.
        # We overlay 1/scale on the same y-axis, normalised to the mean
        # of block 2 across all curves.
        anchor = []
        for cohort, per_block in by_cohort_block.items():
            if 2 in per_block:
                anchor.append(np.mean(per_block[2]))
        if anchor:
            base = float(np.mean(anchor))
            x = np.array([0, 1, 2, 3, 4, 5])
            scale = np.array([1, 1, 1, 2, 2, 2], dtype=float)
            ax.plot(
                x,
                base / scale,
                "k--",
                label="Berrada inverse-upscale (∝1/scale)",
                alpha=0.6,
            )
    ax.set_xlabel("decoder block index")
    ax.set_ylabel(r"$\mathbb{E}\,\|\phi_\ell\|$")
    ax.set_title("Phase 1: per-block feature magnitude")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def channel_concentration_block2_vs_block5(
    rows: Iterable[dict],
    *,
    out_path: Path,
) -> None:
    """Pareto curve: cumulative ``mean_abs`` sorted descending per block."""
    # Aggregate per (block, channel) across patients (mean of mean_abs).
    per_block_chan: dict[int, dict[int, list[float]]] = {}
    for r in rows:
        per_block_chan.setdefault(int(r["block_idx"]), {}).setdefault(
            int(r["channel_idx"]), []
        ).append(float(r["mean_L_dec"]))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for blk in (2, 5):
        if blk not in per_block_chan:
            continue
        chan_means = [float(np.mean(per_block_chan[blk][c])) for c in sorted(per_block_chan[blk])]
        sorted_desc = np.sort(chan_means)[::-1]
        cum = np.cumsum(sorted_desc) / max(np.sum(sorted_desc), 1e-12)
        ax.plot(
            np.arange(1, sorted_desc.size + 1) / sorted_desc.size,
            cum,
            label=f"block {blk}",
        )
    ax.set_xlabel("fraction of channels (sorted)")
    ax.set_ylabel("cumulative L_dec share")
    ax.set_title("Phase 1: per-channel concentration")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def separation_per_region(
    rows: Iterable[dict],
    *,
    out_path: Path,
) -> None:
    """X = block_idx, Y = separation, one line per region (median + IQR)."""
    per_block_region: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        per_block_region.setdefault(int(r["block_idx"]), {}).setdefault(r["region"], []).append(
            float(r["sep_dist"])
        )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    blks = sorted(per_block_region)
    for region in ("WT", "notWT", "global"):
        med = []
        q25 = []
        q75 = []
        for blk in blks:
            vals = per_block_region[blk].get(region, [])
            if not vals:
                med.append(0.0)
                q25.append(0.0)
                q75.append(0.0)
                continue
            arr = np.array(vals)
            med.append(float(np.median(arr)))
            q25.append(float(np.quantile(arr, 0.25)))
            q75.append(float(np.quantile(arr, 0.75)))
        ax.plot(blks, med, "o-", label=region)
        ax.fill_between(blks, q25, q75, alpha=0.2)
    ax.set_xlabel("decoder block index")
    ax.set_ylabel(r"$\|\phi(z_{T_{1c}}) - \phi(z_{T_{1\mathrm{pre}}})\|$")
    ax.set_title("Phase 2: pre/post separation per region")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def t_min_knee(
    rows: Iterable[dict],
    *,
    out_path: Path,
    knee_t: float | None = None,
) -> None:
    """X = t, Y = mean feature distance over blocks, knee marked."""
    per_t_block: dict[float, list[float]] = {}
    for r in rows:
        per_t_block.setdefault(float(r["t"]), []).append(float(r["feature_distance_to_target"]))
    ts = sorted(per_t_block)
    means = [float(np.mean(per_t_block[t])) for t in ts]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(ts, means, "o-")
    if knee_t is not None and ts and (min(ts) <= knee_t <= max(ts)):
        ax.axvline(knee_t, color="red", ls="--", label=f"knee t_min={knee_t:.2f}")
        ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|\phi(z_{T_{1c}}) - \phi(\hat x_1(t))\|$")
    ax.set_title("Phase 2: x̂_1 reliability vs t")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def drift_heatmap(
    rows: Iterable[dict],
    *,
    out_path: Path,
) -> None:
    """Patients × variants heatmap; cell colour = max drift across blocks."""
    # Collapse to one row per (patient, variant) — max drift across blocks.
    per_cell: dict[tuple[str, str, str], float] = {}
    per_cell_pass: dict[tuple[str, str, str], bool] = {}
    for r in rows:
        key = (r["cohort"], r["patient_id"], r["variant"])
        per_cell[key] = max(per_cell.get(key, 0.0), float(r["drift_value"]))
        per_cell_pass[key] = per_cell_pass.get(key, True) and bool(r["passes_gate"])
    patients = sorted({(c, p) for (c, p, _v) in per_cell})
    variants = sorted({v for (_c, _p, v) in per_cell})
    if not patients or not variants:
        return
    M = np.zeros((len(patients), len(variants)))
    fails: list[tuple[int, int]] = []
    for i, (c, p) in enumerate(patients):
        for j, v in enumerate(variants):
            M[i, j] = per_cell.get((c, p, v), 0.0)
            if not per_cell_pass.get((c, p, v), True):
                fails.append((i, j))
    fig, ax = plt.subplots(figsize=(0.45 * len(variants) + 2.0, 0.20 * len(patients) + 2.0))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label="max drift across blocks")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_yticks(range(len(patients)))
    ax.set_yticklabels([f"{c}/{p[:14]}" for (c, p) in patients], fontsize=6)
    for i, j in fails:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", lw=1.3))
    ax.set_title("Phase 3: drift heatmap (cyan box = gate fail)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def inter_cohort_ratio_box(
    rows: Iterable[dict],
    *,
    out_path: Path,
) -> None:
    """Boxplot of per-cohort W/nW ratios (one box per cohort)."""
    per_cohort: dict[str, list[float]] = {}
    for r in rows:
        per_cohort.setdefault(r["cohort"], []).append(float(r["ratio_median"]))
    if not per_cohort:
        return
    cohorts = sorted(per_cohort)
    data = [per_cohort[c] for c in cohorts]
    fig, ax = plt.subplots(figsize=(0.6 * len(cohorts) + 2.0, 4.0))
    ax.boxplot(data, labels=cohorts, vert=True)
    ax.set_ylabel("W/nW ratio at block 5")
    ax.set_title("Phase 3: inter-cohort ratio spread (§4.7c)")
    for tl in ax.get_xticklabels():
        tl.set_rotation(30)
        tl.set_horizontalalignment("right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


__all__ = [
    "channel_concentration_block2_vs_block5",
    "drift_heatmap",
    "inter_cohort_ratio_box",
    "magnitude_curve",
    "separation_per_region",
    "t_min_knee",
]

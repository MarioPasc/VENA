"""Pareto-style scatter for exhaustive-val PSNR × SSIM over epochs and NFE.

Each dot is one `(epoch, nfe)` aggregate (mean across patients + cohorts);
error bars are the per-axis std. Within an epoch, points are sorted by
NFE ascending and joined with a step-post staircase to make the NFE sweep
visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vena.model.fm.post_train.plotting_styles import (
    PAUL_TOL_BRIGHT,
    PAUL_TOL_MUTED,
    PLOT_SETTINGS,
    get_figure_size,
    save_figure,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["aggregate_pareto", "plot_pareto"]


def aggregate_pareto(exhaustive: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Collapse per-patient × NFE rows to per-(epoch, nfe) aggregates.

    Parameters
    ----------
    exhaustive : dict[int, pandas.DataFrame]
        Output of `discover_exhaustive_val`.

    Returns
    -------
    pandas.DataFrame
        Columns: `epoch`, `nfe`, `psnr_mean`, `psnr_std`, `ssim_mean`,
        `ssim_std`, `n_patients`. Sorted by `epoch`, `nfe`.
    """
    import pandas as pd

    rows: list[dict[str, float | int]] = []
    for epoch, df in exhaustive.items():
        if "nfe" not in df.columns:
            continue
        df = df.copy()
        df["epoch"] = epoch
        grouped = df.groupby("nfe", as_index=False).agg(
            psnr_mean=("psnr_db", "mean"),
            psnr_std=("psnr_db", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            n_patients=("psnr_db", "count"),
        )
        for _, r in grouped.iterrows():
            rows.append(
                {
                    "epoch": epoch,
                    "nfe": int(r["nfe"]),
                    "psnr_mean": float(r["psnr_mean"]),
                    "psnr_std": float(r["psnr_std"]) if np.isfinite(r["psnr_std"]) else 0.0,
                    "ssim_mean": float(r["ssim_mean"]),
                    "ssim_std": float(r["ssim_std"]) if np.isfinite(r["ssim_std"]) else 0.0,
                    "n_patients": int(r["n_patients"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["epoch", "nfe", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "n_patients"]
        )
    out = pd.DataFrame(rows).sort_values(["epoch", "nfe"]).reset_index(drop=True)
    return out


def _epoch_size_scale(epoch: int, epoch_min: int, epoch_max: int) -> float:
    """Map an epoch index to a marker size between size_min and size_max."""
    size_min, size_max = 20.0, 110.0
    if epoch_max == epoch_min:
        return (size_min + size_max) / 2.0
    t = (epoch - epoch_min) / (epoch_max - epoch_min)
    return size_min + t * (size_max - size_min)


def _nfe_color_map(nfes: list[int]) -> dict[int, str]:
    return {nfe: PAUL_TOL_MUTED[i % len(PAUL_TOL_MUTED)] for i, nfe in enumerate(nfes)}


def _step_post_segments(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a step-post (horizontal-then-vertical) polyline through points.

    Given `xs = [x0, x1, ..., xk]` and `ys = [y0, ..., yk]` the result has
    `2k + 1` points: each consecutive pair `(xi, yi) -> (x_{i+1}, y_{i+1})`
    becomes `(xi, yi) -> (x_{i+1}, yi) -> (x_{i+1}, y_{i+1})`.
    """
    if xs.size == 0:
        return xs, ys
    sx = [xs[0]]
    sy = [ys[0]]
    for i in range(1, xs.size):
        sx.append(xs[i])
        sy.append(ys[i - 1])
        sx.append(xs[i])
        sy.append(ys[i])
    return np.asarray(sx), np.asarray(sy)


def plot_pareto(exhaustive: dict[int, pd.DataFrame], out_path: Path) -> Path:
    """Pareto plot — PSNR (x) × SSIM (y), one dot per (epoch, nfe).

    Parameters
    ----------
    exhaustive : dict[int, pandas.DataFrame]
        Output of `discover_exhaustive_val`.
    out_path : Path
        Output PNG path. Parent directory created if absent.

    Raises
    ------
    ValueError
        No exhaustive-val data to plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    agg = aggregate_pareto(exhaustive)
    if agg.empty:
        raise ValueError("no exhaustive-val data available; nothing to plot")

    epochs_sorted = sorted(agg["epoch"].unique())
    epoch_min, epoch_max = int(epochs_sorted[0]), int(epochs_sorted[-1])
    nfes_sorted = sorted(agg["nfe"].unique())
    nfe_color = _nfe_color_map(list(map(int, nfes_sorted)))
    guide_color = PAUL_TOL_BRIGHT["grey"]

    fig, ax = plt.subplots(figsize=get_figure_size("double", height_ratio=0.7))

    for epoch in epochs_sorted:
        sub = agg[agg["epoch"] == epoch].sort_values("nfe")
        xs = sub["psnr_mean"].to_numpy()
        ys = sub["ssim_mean"].to_numpy()
        if xs.size >= 2:
            sx, sy = _step_post_segments(xs, ys)
            ax.plot(sx, sy, color=guide_color, alpha=0.4, linewidth=0.9, zorder=1)

        for _, row in sub.iterrows():
            size = _epoch_size_scale(int(row["epoch"]), epoch_min, epoch_max)
            color = nfe_color[int(row["nfe"])]
            ax.errorbar(
                row["psnr_mean"],
                row["ssim_mean"],
                xerr=row["psnr_std"],
                yerr=row["ssim_std"],
                fmt="none",
                ecolor=color,
                elinewidth=PLOT_SETTINGS["errorbar_linewidth"],
                capsize=PLOT_SETTINGS["errorbar_capsize"],
                capthick=PLOT_SETTINGS["errorbar_capthick"],
                alpha=0.7,
                zorder=2,
            )
            ax.scatter(
                row["psnr_mean"],
                row["ssim_mean"],
                s=size,
                c=color,
                edgecolors="white",
                linewidths=PLOT_SETTINGS["scatter_edgewidth"],
                alpha=0.9,
                zorder=3,
            )

    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("SSIM")
    ax.set_title("PSNR-SSIM trade-off across epochs and NFE")

    nfe_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=nfe_color[int(nfe)],
            markeredgecolor="white",
            markersize=8,
            label=f"NFE={nfe}",
        )
        for nfe in nfes_sorted
    ]
    median_epoch = int(epochs_sorted[len(epochs_sorted) // 2])
    epoch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PAUL_TOL_BRIGHT["grey"],
            markeredgecolor="white",
            markersize=np.sqrt(_epoch_size_scale(ep, epoch_min, epoch_max)),
            label=f"epoch {ep}",
        )
        for ep in [epoch_min, median_epoch, epoch_max]
    ]
    handles = nfe_handles + epoch_handles
    ncol = min(max(len(handles), 1), 6)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        frameon=PLOT_SETTINGS["legend_frameon"],
        fontsize=PLOT_SETTINGS["legend_fontsize"],
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_path.with_suffix("")), formats=(out_path.suffix.lstrip("."),))
    plt.close(fig)
    return out_path

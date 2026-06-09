"""Loss + grad-norm plots for `train_epoch.csv`.

Two figures are produced:

* `plot_total_grad(df, out_path)` — total loss with `cfm` / `contrastive`
  decomposition on `y1`, postclip grad norms on `y2`.
* `plot_per_cohort_grad(df, out_path)` — one CFM line per active cohort on
  `y1`, postclip grad norms on `y2`.

Both functions are pure: they create a matplotlib figure, write it to disk
through `save_figure`, close it, and return the saved path. They do not
mutate the input DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vena.model.fm.post_train.loaders import (
    active_grad_norm_series,
    detect_active_cohorts,
    detect_active_losses,
)
from vena.model.fm.post_train.plotting_styles import (
    PAUL_TOL_BRIGHT,
    PAUL_TOL_HIGH_CONTRAST,
    PAUL_TOL_MUTED,
    PLOT_SETTINGS,
    get_figure_size,
    save_figure,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["plot_per_cohort_grad", "plot_total_grad"]

_LOSS_COLOR = {
    "cfm": PAUL_TOL_BRIGHT["blue"],
    "contrastive": PAUL_TOL_BRIGHT["red"],
    "total": PAUL_TOL_BRIGHT["purple"],
}

_GRAD_COLOR = {
    "cn": PAUL_TOL_HIGH_CONTRAST["yellow"],
    "trunk": PAUL_TOL_HIGH_CONTRAST["red"],
}


def _plot_series_with_band(ax, x, mean, std, *, color, label: str, alpha_line: float) -> None:
    """Draw a `mean` line and a `mean ± std` band on `ax`."""
    import numpy as np

    band_alpha = PLOT_SETTINGS["error_band_alpha"]
    line_width = PLOT_SETTINGS["line_width"]

    mean_arr = np.asarray(mean, dtype=float)
    std_arr = np.asarray(std, dtype=float)
    finite = np.isfinite(mean_arr)
    if not finite.any():
        return
    ax.plot(
        x[finite],
        mean_arr[finite],
        color=color,
        alpha=alpha_line,
        linewidth=line_width,
        label=label,
    )
    if std_arr.shape == mean_arr.shape and np.any(np.isfinite(std_arr)):
        valid = finite & np.isfinite(std_arr)
        ax.fill_between(
            x[valid],
            mean_arr[valid] - std_arr[valid],
            mean_arr[valid] + std_arr[valid],
            color=color,
            alpha=band_alpha,
            linewidth=0,
        )


def _add_grad_norm_axis(ax, df: pd.DataFrame) -> object:
    """Create the `y2` axis and overlay postclip grad-norm series.

    Returns the twin axis so the caller can collect legend handles.
    """
    branches = active_grad_norm_series(df)
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel("Grad norm (postclip)")
    epochs = df["epoch"].to_numpy()
    for branch in branches:
        _plot_series_with_band(
            ax2,
            epochs,
            df[f"grad_norm_{branch}_postclip_mean"],
            df.get(f"grad_norm_{branch}_postclip_std"),
            color=_GRAD_COLOR[branch],
            label=f"grad-norm {branch}",
            alpha_line=0.85,
        )
    ax2.grid(False)
    return ax2


def _outside_legend(fig, *axes) -> None:
    """Collect legend entries from `axes` and place a combined legend below."""
    handles: list = []
    labels: list[str] = []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)
    if not handles:
        return
    ncol = min(max(len(handles), 1), 4)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        frameon=PLOT_SETTINGS["legend_frameon"],
        fontsize=PLOT_SETTINGS["legend_fontsize"],
    )


def plot_total_grad(df: pd.DataFrame, out_path: Path) -> Path:
    """Plot 1A — total loss + cfm/contrastive decomposition + grad norms.

    Parameters
    ----------
    df : pandas.DataFrame
        From `load_train_epoch_csv`.
    out_path : Path
        Output PNG path. Parent directory is created if absent.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=get_figure_size("double", height_ratio=0.45))
    epochs = df["epoch"].to_numpy()
    active = detect_active_losses(df)

    # If only cfm is active (S1), draw the cfm line solo without a duplicate
    # `total` overlay. Otherwise: decomposition at α=0.6 + total at α=1.0.
    if active == ["cfm"]:
        _plot_series_with_band(
            ax,
            epochs,
            df["cfm_mean"],
            df.get("cfm_std"),
            color=_LOSS_COLOR["cfm"],
            label="CFM loss",
            alpha_line=1.0,
        )
    else:
        for name in active:
            _plot_series_with_band(
                ax,
                epochs,
                df[f"{name}_mean"],
                df.get(f"{name}_std"),
                color=_LOSS_COLOR[name],
                label=f"{name} loss",
                alpha_line=0.6,
            )
        if "total_mean" in df.columns:
            _plot_series_with_band(
                ax,
                epochs,
                df["total_mean"],
                df.get("total_std"),
                color=_LOSS_COLOR["total"],
                label="total loss",
                alpha_line=1.0,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss and gradient norm")
    ax2 = _add_grad_norm_axis(ax, df)

    _outside_legend(fig, ax, ax2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_path.with_suffix("")), formats=(out_path.suffix.lstrip("."),))
    plt.close(fig)
    return out_path


def plot_per_cohort_grad(df: pd.DataFrame, out_path: Path) -> Path:
    """Plot 1B — per-cohort CFM loss + grad norms.

    Parameters
    ----------
    df : pandas.DataFrame
        From `load_train_epoch_csv`.
    out_path : Path
        Output PNG path. Parent directory is created if absent.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=get_figure_size("double", height_ratio=0.5))
    epochs = df["epoch"].to_numpy()
    cohorts = detect_active_cohorts(df, loss="cfm")
    palette = PAUL_TOL_MUTED

    for idx, cohort in enumerate(cohorts):
        color = palette[idx % len(palette)]
        _plot_series_with_band(
            ax,
            epochs,
            df[f"cfm_cohort_{cohort}_mean"],
            df.get(f"cfm_cohort_{cohort}_std"),
            color=color,
            label=cohort,
            alpha_line=0.6,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-cohort CFM loss")
    ax.set_title("Per-cohort CFM loss and gradient norm")
    ax2 = _add_grad_norm_axis(ax, df)

    _outside_legend(fig, ax, ax2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_path.with_suffix("")), formats=(out_path.suffix.lstrip("."),))
    plt.close(fig)
    return out_path

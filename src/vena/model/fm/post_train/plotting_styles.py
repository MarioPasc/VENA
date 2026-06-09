"""Publication-ready plot settings and utilities for IEEE-compliant figures.

IEEE-compliant settings with Paul Tol colorblind-friendly palettes.

References:
    - Paul Tol's color schemes: https://personal.sron.nl/~pault/
    - IEEE publication guidelines
    - scienceplots: https://github.com/garrettj403/SciencePlots
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "IEEE_COLUMN_GAP_INCHES",
    "IEEE_COLUMN_WIDTH_INCHES",
    "IEEE_TEXT_HEIGHT_INCHES",
    "IEEE_TEXT_WIDTH_INCHES",
    "PAUL_TOL_BRIGHT",
    "PAUL_TOL_HIGH_CONTRAST",
    "PAUL_TOL_MUTED",
    "PLOT_SETTINGS",
    "apply_plot_settings",
    "get_effect_size_interpretation",
    "get_figure_size",
    "get_significance_stars",
    "save_figure",
    "save_latex_table",
]

# =============================================================================
# Paul Tol Color Palettes (SRON - colorblind safe)
# =============================================================================

PAUL_TOL_BRIGHT: dict[str, str] = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

PAUL_TOL_HIGH_CONTRAST: dict[str, str] = {
    "blue": "#004488",
    "yellow": "#DDAA33",
    "red": "#BB5566",
}

PAUL_TOL_MUTED: list[str] = [
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
]

# =============================================================================
# IEEE Column Width Specifications
# =============================================================================

IEEE_COLUMN_WIDTH_INCHES = 3.39  # Single column (86 mm)
IEEE_COLUMN_GAP_INCHES = 0.24  # Gap between columns (6 mm)
IEEE_TEXT_WIDTH_INCHES = 7.0  # Full print area width (178 mm)
IEEE_TEXT_HEIGHT_INCHES = 9.0  # Full print area height (229 mm)

# =============================================================================
# Main Plot Settings Dictionary
# =============================================================================

PLOT_SETTINGS: dict[str, Any] = {
    # Figure dimensions (IEEE compliant)
    "figure_width_single": IEEE_COLUMN_WIDTH_INCHES,
    "figure_width_double": IEEE_TEXT_WIDTH_INCHES,
    "figure_height_max": IEEE_TEXT_HEIGHT_INCHES,
    "figure_height_ratio": 0.75,
    # Fonts (IEEE requires Times or similar serif)
    "font_family": "serif",
    "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext_fontset": "stix",
    "text_usetex": False,
    # Font sizes
    "font_size": 10,
    "axes_labelsize": 11,
    "axes_titlesize": 12,
    "tick_labelsize": 9,
    "legend_fontsize": 9,
    "annotation_fontsize": 8,
    "panel_label_fontsize": 11,
    # Line properties
    "line_width": 1.2,
    "line_width_thick": 1.8,
    "marker_size": 5,
    "marker_edge_width": 0.5,
    # Error bars
    "errorbar_capsize": 2,
    "errorbar_capthick": 0.8,
    "errorbar_linewidth": 0.8,
    # Error bands (for confidence intervals)
    "error_band_alpha": 0.2,
    # Boxplot properties
    "boxplot_linewidth": 0.8,
    "boxplot_flier_size": 3,
    "boxplot_width": 0.6,
    # Bar plot properties
    "bar_width": 0.18,
    "bar_alpha": 0.85,
    # Grid
    "grid_alpha": 0.4,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    # Spines
    "spine_linewidth": 0.8,
    "spine_color": "0.2",
    # Ticks
    "tick_direction": "in",
    "tick_major_width": 0.8,
    "tick_minor_width": 0.5,
    "tick_major_length": 3.5,
    "tick_minor_length": 2.0,
    # Legend
    "legend_frameon": False,
    "legend_framealpha": 0.9,
    "legend_edgecolor": "0.8",
    "legend_borderpad": 0.4,
    "legend_columnspacing": 1.0,
    "legend_handletextpad": 0.5,
    # Scatter
    "scatter_alpha": 0.6,
    "scatter_size": 15,
    "scatter_edgewidth": 0.3,
    # DPI for output
    "dpi_print": 300,
    "dpi_screen": 150,
    # Significance annotations
    "significance_bracket_linewidth": 0.8,
    "significance_text_fontsize": 9,
    "effect_size_fontsize": 8,
}


_APPLIED = False


def apply_plot_settings() -> None:
    """Apply `PLOT_SETTINGS` to `matplotlib.rcParams`.

    Idempotent — safe to call from multiple entrypoints. The user-supplied
    settings dictionary is translated to the relevant rcParams keys; only
    the keys that map cleanly are forwarded.
    """
    global _APPLIED
    if _APPLIED:
        return
    import matplotlib as mpl

    rc = {
        "figure.dpi": PLOT_SETTINGS["dpi_screen"],
        "savefig.dpi": PLOT_SETTINGS["dpi_print"],
        "font.family": PLOT_SETTINGS["font_family"],
        "font.serif": PLOT_SETTINGS["font_serif"],
        "font.size": PLOT_SETTINGS["font_size"],
        "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],
        "text.usetex": PLOT_SETTINGS["text_usetex"],
        "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
        "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
        "axes.linewidth": PLOT_SETTINGS["spine_linewidth"],
        "axes.edgecolor": PLOT_SETTINGS["spine_color"],
        "axes.grid": True,
        "grid.alpha": PLOT_SETTINGS["grid_alpha"],
        "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
        "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
        "xtick.direction": PLOT_SETTINGS["tick_direction"],
        "ytick.direction": PLOT_SETTINGS["tick_direction"],
        "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
        "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
        "xtick.major.size": PLOT_SETTINGS["tick_major_length"],
        "ytick.major.size": PLOT_SETTINGS["tick_major_length"],
        "lines.linewidth": PLOT_SETTINGS["line_width"],
        "lines.markersize": PLOT_SETTINGS["marker_size"],
        "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
        "legend.frameon": PLOT_SETTINGS["legend_frameon"],
        "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
        "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],
        "legend.borderpad": PLOT_SETTINGS["legend_borderpad"],
        "legend.columnspacing": PLOT_SETTINGS["legend_columnspacing"],
        "legend.handletextpad": PLOT_SETTINGS["legend_handletextpad"],
    }
    mpl.rcParams.update(rc)
    _APPLIED = True


def get_significance_stars(p_val: float) -> str:
    """Convert p-value to significance stars.

    Parameters
    ----------
    p_val : float
        P-value from statistical test.

    Returns
    -------
    str
        "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), or "n.s.".
    """
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "n.s."


def get_effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    if d_abs < 0.5:
        return "small"
    if d_abs < 0.8:
        return "medium"
    return "large"


def get_figure_size(
    width: str = "single",
    height_ratio: float | None = None,
) -> tuple[float, float]:
    """Get figure size tuple for IEEE format.

    Parameters
    ----------
    width : str
        "single" for column width, "double" for full width.
    height_ratio : float | None
        Custom height/width ratio. If None, uses default.
    """
    if width == "single":
        w = PLOT_SETTINGS["figure_width_single"]
    elif width == "double":
        w = PLOT_SETTINGS["figure_width_double"]
    else:
        raise ValueError(f"Unknown width: {width}")

    ratio = height_ratio if height_ratio is not None else PLOT_SETTINGS["figure_height_ratio"]
    return (w, w * ratio)


def save_figure(
    fig: Figure,
    path: str,
    formats: tuple[str, ...] = ("pdf", "png", "svg"),
) -> list[str]:
    """Save figure in multiple formats at publication DPI.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str
        Base path without extension (e.g. "/tmp/fig1").
    formats : tuple[str, ...]
        Format extensions; one file per entry.
    """
    saved: list[str] = []
    for fmt in formats:
        fpath = f"{path}.{fmt}"
        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        fig.savefig(
            fpath,
            format=fmt,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
            pad_inches=0.02,
        )
        saved.append(fpath)
    return saved


def save_latex_table(
    df: Any,
    path: str,
    caption: str,
    label: str,
) -> None:
    """Save a pandas DataFrame as a LaTeX table."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    latex = df.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        position="htbp",
    )
    with open(path, "w") as f:
        f.write(latex)

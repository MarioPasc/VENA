"""Phase-2 shared figure substrate.

Provides :func:`setup_figure_style`, :func:`method_keyed_renderer`,
:func:`annotate_significance`, :func:`method_palette`, and
:func:`method_order`.

All three Phase-2 routines (paired fidelity §4.2, spatial residual §4.3,
downstream seg §4.4) import their significance annotation and palette from
here so that figures are visually consistent across the paper.

``render_comparison_figure`` in ``vena.model.fm.eval.exhaustive`` is **not
modified** — it is keyed by NFE and guarded by a unit test that forbids
signature changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from vena.validation.stats import HolmResult


def setup_figure_style() -> None:
    """Configure matplotlib rcParams for the Phase-2 house style.

    Sets black figure/axes facecolor, gray colormap default, and constrained
    layout.  Call once per session before creating any Phase-2 figure.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "image.cmap": "gray",
            "figure.constrained_layout.use": True,
        }
    )


def method_palette() -> dict[str, str]:
    """Return the canonical colourblind-safe palette for all 16 methods.

    Delegates to :func:`vena.validation.registry.method_palette` so callers
    do not need to import the registry directly.

    Returns
    -------
    dict[str, str]
        ``{method_key: "#RRGGBB"}``.
    """
    from vena.validation.registry import method_palette as _palette

    return _palette()


def method_order() -> list[str]:
    """Return the canonical display order for all pre-registered methods.

    Delegates to :func:`vena.validation.registry.method_order`.

    Returns
    -------
    list[str]
        Method keys in canonical display order (VENA → family → ablation →
        supplementary).
    """
    from vena.validation.registry import method_order as _order

    return _order()


def annotate_significance(
    ax: matplotlib.axes.Axes,
    pairs: list[tuple[str, str]],
    holm_results: dict[str, HolmResult],
    *,
    x_positions: dict[str, float],
    y_position: float | None = None,
    fontsize: int = 6,
) -> None:
    """Draw Holm-corrected significance markers on *ax*.

    Places a star label (``***`` / ``**`` / ``*`` / ``ns``) at each
    competitor's x position.  Use for bar charts that compare each competitor
    to VENA at a fixed y level.

    Parameters
    ----------
    ax :
        Axes on which to draw.  Must already contain the bars whose positions
        are referenced by *x_positions*.
    pairs :
        List of ``(vena_key, competitor_key)`` tuples.  Only the competitor
        key is used for look-ups in *holm_results* and *x_positions*; the
        VENA key is accepted for forward-compatibility with multi-reference
        designs.
    holm_results :
        ``{competitor_key: HolmResult}`` as returned by
        :func:`vena.validation.stats.holm_bonferroni`.  Keys that are absent
        from this dict are silently skipped.
    x_positions :
        ``{method_key: float}`` mapping each competitor to its bar centre on
        the x-axis.  Keys not in *pairs* are ignored.
    y_position :
        Fixed y-coordinate for all markers.  If ``None`` (default), uses
        95 % of the current axes y-limit upper bound.
    fontsize :
        Font size for the significance label.

    Notes
    -----
    Significance thresholds (Bonferroni-corrected p-values after Holm step):
    - ``***`` : p_adj < 0.001
    - ``**``  : p_adj < 0.01
    - ``*``   : p_adj < 0.05 (reject == True)
    - ``ns``  : p_adj ≥ 0.05 (reject == False)
    """
    if y_position is None:
        ylo, yhi = ax.get_ylim()
        y_position = ylo + 0.95 * (yhi - ylo)

    for _vena_key, competitor in pairs:
        result = holm_results.get(competitor)
        if result is None:
            continue
        x = x_positions.get(competitor)
        if x is None:
            continue

        p = result.pvalue_adj
        if result.reject:
            if p < 0.001:
                label = "***"
            elif p < 0.01:
                label = "**"
            else:
                label = "*"
        else:
            label = "ns"

        ax.text(
            x,
            y_position,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="white",
            clip_on=False,
        )


def method_keyed_renderer(
    real_vol: np.ndarray,
    synth_by_method: dict[str, np.ndarray],
    slice_indices: list[int],
    *,
    tag: str = "",
    patient_id: str = "",
    metric_by_method: dict[str, dict[str, float]] | None = None,
) -> matplotlib.figure.Figure:
    """Render a black-background comparison figure keyed by method.

    Analogous to ``render_comparison_figure`` in
    ``vena.model.fm.eval.exhaustive`` but the rows are methods rather than NFE
    levels.  Rows are sorted by the primary metric (SSIM by default) descending
    so the best synthesis is immediately below the reference row.

    Do **not** call ``render_comparison_figure`` — it is keyed by NFE and
    guarded by a unit test that forbids signature changes.

    Parameters
    ----------
    real_vol :
        ``(H, W, D)`` float32 reference volume.
    synth_by_method :
        ``{method_name: (H, W, D) float32 synthetic volume}``.
    slice_indices :
        Axial slice indices to display (from
        ``vena.model.fm.eval.exhaustive.select_content_slices``).
    tag :
        Short run tag for the suptitle.
    patient_id :
        Patient identifier for the suptitle.
    metric_by_method :
        Optional ``{method: {"ssim": ..., "psnr": ...}}`` for row ylabels.

    Returns
    -------
    matplotlib.figure.Figure
        Black-background figure ready for ``savefig``.
    """
    import matplotlib.pyplot as plt

    n_methods = len(synth_by_method)
    n_slices = len(slice_indices)
    n_rows = 1 + n_methods  # real + one per method

    # Sort methods by SSIM descending (best first), fallback to insertion order.
    if metric_by_method:
        sorted_methods = sorted(
            synth_by_method.keys(),
            key=lambda m: metric_by_method.get(m, {}).get("ssim", -1.0),
            reverse=True,
        )
    else:
        sorted_methods = list(synth_by_method.keys())

    fig, axes = plt.subplots(
        n_rows,
        n_slices,
        figsize=(n_slices * 2.0, n_rows * 2.0),
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    def _show_row(row_axes: list, vol: np.ndarray, label: str) -> None:
        for ax, k in zip(row_axes, slice_indices, strict=False):
            # Per-slice intensity window anchored to the real slice's (min, max).
            real_slice = real_vol[..., k]
            vmin = float(real_slice.min())
            vmax = float(real_slice.max())
            if vmax <= vmin:
                vmin, vmax = 0.0, 1.0
            ax.imshow(
                vol[..., k].T,
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(label, color="white", fontsize=7)

    # Row 0: real T1c
    _show_row(list(axes[0]), real_vol, "Real T1c")

    # Method rows
    for row_idx, method in enumerate(sorted_methods, start=1):
        vol = synth_by_method[method]
        if metric_by_method and method in metric_by_method:
            m = metric_by_method[method]
            label = (
                f"{method}\n"
                f"SSIM={m.get('ssim', float('nan')):.3f} / "
                f"PSNR={m.get('psnr', float('nan')):.1f}dB"
            )
        else:
            label = method
        _show_row(list(axes[row_idx]), vol, label)

    suptitle = f"{tag} — {patient_id}" if tag or patient_id else ""
    if suptitle:
        fig.suptitle(suptitle, color="white", fontsize=9)

    return fig

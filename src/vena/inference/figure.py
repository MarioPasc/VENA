"""Per-cohort multi-method comparison figure.

For every test cohort the engine renders one PNG showing the same
patient predicted by every benchmarked method (one row per method) at
the method's §5.1 selection NFE, with the real T1c as the top row, at
``n_slices`` equally-spaced axial slices selected via
:func:`vena.model.fm.eval.exhaustive.select_content_slices`.

This is the figure the user pointed at as the smoke-run acceptance
criterion ("1 PNG per dataset, ... same patient being predicted by the
different models ... along with the ground truth"). The protocol's
qualitative-reporting expectation (§9 failure-mode taxonomy) reuses the
same layout for the final paper figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from vena.model.fm.eval import select_content_slices


def render_multi_method_figure(
    *,
    cohort: str,
    patient_id: str,
    real_t1c: torch.Tensor | np.ndarray,
    method_predictions: Sequence[tuple[str, torch.Tensor | np.ndarray, int, float]],
    out_path: Path | str,
    n_slices: int = 7,
    slice_offset: int = 10,
    title_suffix: str | None = None,
) -> Path:
    """Render the (1 + n_methods) × n_slices comparison panel.

    Parameters
    ----------
    cohort, patient_id
        Used in the figure title only.
    real_t1c
        ``(H, W, D)`` reference volume, already §4.1 harmonised
        (``[0, 1]`` over brain mask).
    method_predictions
        Sequence of ``(method_name, predicted_volume, nfe, seconds)``
        tuples, **in display order** (one row per tuple, in the order
        given). ``predicted_volume`` is the §4.1 harmonised prediction
        (``(H, W, D)`` in ``[0, 1]``).
    out_path
        PNG destination. Parent directories are created.
    n_slices
        Number of axial slices (columns). Default 7 matches the smoke
        acceptance criterion.
    slice_offset
        Inward shrink applied to the content range; forwarded to
        :func:`select_content_slices`.
    title_suffix
        Optional extra text appended to the suptitle (e.g. ``"smoke"``).

    Returns
    -------
    Path
        ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _np(v: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().float().numpy()
        return np.asarray(v, dtype=np.float32)

    real_np = _np(real_t1c)
    slice_indices = select_content_slices(real_np, n_slices=n_slices, offset=slice_offset)

    rows = [("Real T1c", real_np, None, None)] + [
        (name, _np(vol), nfe, seconds) for name, vol, nfe, seconds in method_predictions
    ]
    n_rows, n_cols = len(rows), len(slice_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 * n_cols, 1.5 * n_rows), squeeze=False)
    for r_idx, (label, vol, nfe, seconds) in enumerate(rows):
        if nfe is None or seconds is None:
            row_label = label
        else:
            row_label = f"{label}\nNFE={int(nfe)} (t={float(seconds):.2f}s)"
        for c_idx, k in enumerate(slice_indices):
            ax = axes[r_idx][c_idx]
            ax.imshow(np.rot90(vol[..., k]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(f"z={k}", fontsize=7)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=7)

    title = f"{cohort} — {patient_id}"
    if title_suffix:
        title = f"{title}  [{title_suffix}]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = ["render_multi_method_figure"]

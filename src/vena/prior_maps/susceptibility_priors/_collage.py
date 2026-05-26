"""Per-patient QC collage for susceptibility-prior outputs.

Same 3 + 1 + 3 layout as the other prior modules. Source column = SWI (the
bias-corrected SWAN magnitude); middle column = ``sus`` channel; right group =
SWI + ``sus`` overlay + threshold contour.

Delegates to the perfusion module's parameterised :func:`render_collage` to
avoid duplicating matplotlib code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vena.prior_maps.perfusion_priors._collage import render_collage as _render_collage


def render_collage(
    source: NDArray[Any],
    brain: NDArray[Any],
    channel: NDArray[np.float32],
    binary: NDArray[np.uint8] | None,
    out_path: Path,
    *,
    patient_id: str,
    channel_vmax: float = 1.0,
    n_slices: int = 5,
    dpi: int = 150,
    overlay_alpha: float = 0.7,
    min_voxels_per_slice: int = 2000,
) -> Path:
    """Render the susceptibility collage. See module docstring for layout."""
    return _render_collage(
        source=source,
        brain=brain,
        channel=channel,
        binary=binary,
        out_path=out_path,
        patient_id=patient_id,
        source_label="SWI",
        channel_label="sus (darkness field)",
        channel_vmin=0.0,
        channel_vmax=channel_vmax,
        n_slices=n_slices,
        dpi=dpi,
        overlay_alpha=overlay_alpha,
        min_voxels_per_slice=min_voxels_per_slice,
    )

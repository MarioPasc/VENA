"""Per-patient QC collage for cellularity-prior outputs.

Same 3 + 1 + 3 layout as :mod:`vena.prior_maps.vessel_priors._collage` and
:mod:`vena.prior_maps.perfusion_priors._collage`. The source column shows ADC,
the middle column shows the primary ``cell`` channel (range ``[0, 1]``), and
the right group shows ADC + ``cell`` overlay with the tumour-mask contour.

Implementation note: the per-module collage delegates to the perfusion module's
:func:`render_collage` helper, which is module-agnostic by construction
(parameterised on source, channel, label, vmin/vmax). Keeping a per-module file
preserves the layout fixed by ``.claude/rules/preflight-pattern.md`` while
avoiding the maintenance burden of duplicated matplotlib code.
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
    n_slices: int = 5,
    dpi: int = 150,
    overlay_alpha: float = 0.7,
    min_voxels_per_slice: int = 2000,
) -> Path:
    """Render the cellularity collage. See module docstring for layout."""
    return _render_collage(
        source=source,
        brain=brain,
        channel=channel,
        binary=binary,
        out_path=out_path,
        patient_id=patient_id,
        source_label="ADC",
        channel_label="cell (tumour-restricted)",
        channel_vmin=0.0,
        channel_vmax=1.0,
        n_slices=n_slices,
        dpi=dpi,
        overlay_alpha=overlay_alpha,
        min_voxels_per_slice=min_voxels_per_slice,
    )

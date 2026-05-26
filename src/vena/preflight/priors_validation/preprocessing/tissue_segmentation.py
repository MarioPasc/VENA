"""NAWM and ventricle masks built from the cohort's bundled segmentations.

UCSF-PDGM ships ``brain_segmentation`` (binary brain) and
``brain_parenchyma_segmentation`` (WM+GM). We use those directly:

* **NAWM** = atlas-warped white-matter ROI (Harvard-Oxford "Cerebral White
  Matter") ∩ subject parenchyma ∖ subject tumour. Without an explicit white-
  matter segmentation (FSL FAST is out-of-scope for v0), this two-source
  intersection is the cleanest proxy.
* **Ventricles** = brain ∖ parenchyma, optionally refined by an atlas-warped
  lateral-ventricle ROI to remove sulcal CSF.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_nawm_mask(
    parenchyma_mask: NDArray[np.integer] | NDArray[np.bool_] | None,
    tumour_mask: NDArray[np.integer] | NDArray[np.bool_] | None,
    atlas_wm_mask: NDArray[np.integer] | NDArray[np.bool_] | None,
    brain_mask: NDArray[np.integer] | NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Construct a NAWM proxy from the available masks.

    Falls back to ``parenchyma ∖ tumour`` when the atlas WM mask is absent
    (atlas registration failed), and to ``brain ∖ tumour`` if parenchyma is
    also absent (extremely defensive — the routine usually has both).
    """
    brain = np.asarray(brain_mask) > 0
    nawm = brain.copy()
    if atlas_wm_mask is not None:
        nawm &= np.asarray(atlas_wm_mask) > 0
    if parenchyma_mask is not None:
        nawm &= np.asarray(parenchyma_mask) > 0
    if tumour_mask is not None:
        nawm &= ~(np.asarray(tumour_mask) > 0)
    return nawm


def build_ventricle_mask(
    parenchyma_mask: NDArray[np.integer] | NDArray[np.bool_] | None,
    brain_mask: NDArray[np.integer] | NDArray[np.bool_],
    atlas_ventricle_mask: NDArray[np.integer] | NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Ventricle / CSF proxy: brain ∖ parenchyma; optionally intersected
    with an atlas-warped lateral-ventricle ROI."""
    brain = np.asarray(brain_mask) > 0
    if parenchyma_mask is not None:
        csf = brain & (~(np.asarray(parenchyma_mask) > 0))
    else:
        csf = brain.copy()
    if atlas_ventricle_mask is not None:
        csf &= np.asarray(atlas_ventricle_mask) > 0
    return csf

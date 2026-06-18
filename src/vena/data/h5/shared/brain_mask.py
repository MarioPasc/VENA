"""Brain-mask post-processing shared across cohort converters.

The cohort image-H5 converters derive ``masks/brain`` either from a shipped
NIfTI mask (UCSF-PDGM, LUMIERE) or from a per-modality nonzero union after
skull-stripping (BraTS-GLI, BraTS-Africa, BraTS-PED, IvyGAP, UPENN-GBM,
REMBRANDT). The latter produces, on a non-trivial fraction of patients,
many small spurious connected components from intensity jitter at the brain
boundary — see ``.claude/notes/data/2026-06-18_data_audit.md`` for the
per-cohort empirical breakdown (IvyGAP 35–148 CCs / sample; others mild).

This module supplies ``clean_brain_mask`` to drop sub-threshold CCs before
the converter writes ``masks/brain`` to the H5. It keeps every component of
volume at least ``min_component_voxels``, so the cerebellum, brainstem, and
detached white-matter pockets are preserved — only the boundary noise is
dropped. The threshold is intentionally conservative: at 1 mm isotropic
voxel spacing, 1000 voxels ≈ 1 cm³, well below any real anatomic region.

Use from cohort converters as::

    from vena.data.h5.shared.brain_mask import clean_brain_mask

    brain = clean_brain_mask(brain_raw, min_component_voxels=1000)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import label

logger = logging.getLogger(__name__)

_DEFAULT_MIN_COMPONENT_VOXELS = 1000


def clean_brain_mask(
    mask: np.ndarray,
    *,
    min_component_voxels: int = _DEFAULT_MIN_COMPONENT_VOXELS,
    structure: np.ndarray | None = None,
) -> np.ndarray:
    """Drop connected components smaller than ``min_component_voxels``.

    Preserves every CC at or above the threshold so multi-region brains
    (cerebellum + cerebrum + brainstem after skull-strip) survive intact.

    Parameters
    ----------
    mask : np.ndarray
        Binary brain mask, shape ``(H, W, D)``. Any non-zero dtype accepted;
        the comparison is ``mask > 0``.
    min_component_voxels : int
        Inclusive lower bound on CC volume. Defaults to ``1000`` (~1 cm³ at
        1 mm³ voxels), which is below the volume of any real brain region.
    structure : np.ndarray | None
        Connectivity structuring element passed to ``scipy.ndimage.label``.
        ``None`` (default) uses 6-connectivity (face neighbours), which is
        the safest choice for soft brain masks — 26-connectivity tends to
        bridge boundary noise across the skull-strip seam.

    Returns
    -------
    np.ndarray
        Same shape and dtype as ``mask``. Voxels in dropped CCs are set to
        ``0``; survivors keep their original value.
    """
    if mask.ndim != 3:
        raise ValueError(f"clean_brain_mask expects 3-D input; got shape {mask.shape}")
    if min_component_voxels <= 0:
        raise ValueError(f"min_component_voxels must be positive; got {min_component_voxels}")

    binary = mask > 0
    if not binary.any():
        logger.warning("clean_brain_mask: input mask is all-zero; passing through")
        return mask

    labelled, n_components = label(binary, structure=structure)
    if n_components <= 1:
        return mask

    # bincount[0] = background voxels; CCs are 1..n_components.
    sizes = np.bincount(labelled.ravel())
    cc_sizes = sizes[1:]  # exclude background
    kept_ids = np.flatnonzero(cc_sizes >= min_component_voxels) + 1  # +1 → CC label
    if kept_ids.size == 0:
        # Pathological case: every CC is below threshold. Keep the largest
        # one so we never return an empty mask — log loudly so the encode
        # routine can flag the patient.
        biggest = int(np.argmax(cc_sizes)) + 1
        kept_ids = np.array([biggest], dtype=np.int64)
        logger.warning(
            "clean_brain_mask: every CC below %d voxels; keeping the largest (size=%d)",
            min_component_voxels,
            int(cc_sizes[biggest - 1]),
        )

    survivors = np.isin(labelled, kept_ids)
    dropped_voxels = int(binary.sum() - survivors.sum())
    if dropped_voxels > 0:
        logger.debug(
            "clean_brain_mask: dropped %d/%d CCs (%d voxels)",
            int(n_components - kept_ids.size),
            int(n_components),
            dropped_voxels,
        )

    out = mask.copy()
    out[~survivors] = 0
    return out

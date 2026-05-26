"""Helpers + per-subject test context shared by the five tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.dataclasses import SubjectInputs


@dataclass(frozen=True)
class TestContext:
    """Per-subject pre-computed context handed to every test.

    The runner builds this once per subject (atlas warp + tissue masks +
    enhancement map) and threads it through T1…T5 so each test does not have
    to recompute the heavy bits.
    """

    subject: SubjectInputs
    # Atlas-warped subject-space label maps keyed by atlas_id from registry.py
    atlas_labels: dict[str, NDArray[np.int32]] = field(default_factory=dict)
    # Pre-built tissue masks (NAWM, ventricles) — bool arrays in subject space
    nawm_mask: NDArray[np.bool_] | None = None
    ventricle_mask: NDArray[np.bool_] | None = None
    # Pre-computed enhancement map ΔT1 = zscore(T1c) − zscore(T1pre), float32
    delta_t1: NDArray[np.floating] | None = None
    # Registration diagnostic — Dice between warped ventricle ROI and
    # subject's CSF proxy. Set by the runner.
    atlas_registration_dice: float | None = None
    # Tags propagated to outcomes (e.g. extra warnings)
    warnings: tuple[str, ...] = ()


def roi_mask_from_atlas(
    atlas_labels: NDArray[np.int32],
    label_values: tuple[int, ...],
    extra_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Build a binary ROI mask from one atlas-warped label image."""
    out = np.isin(atlas_labels, np.asarray(label_values, dtype=np.int32))
    if extra_mask is not None:
        out &= np.asarray(extra_mask) > 0
    return out


def safe_median(values: NDArray[np.floating], mask: NDArray[np.bool_]) -> float:
    """Median over masked voxels; NaN if mask is empty."""
    m = np.asarray(mask) > 0
    if not m.any():
        return float("nan")
    return float(np.median(values[m]))

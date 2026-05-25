"""Analysis primitives for the vessel-mask preflight.

All functions take numpy arrays (no ITK / nibabel dependencies) so they are
unit-testable in isolation. The engine in :mod:`engine` wires them together.

Metric definitions
------------------
* **Binary fraction** — voxels above the threshold within the brain mask,
  divided by total brain voxels.
* **Connected components** — 3D 26-connected (``scipy.ndimage.label``) blobs
  in the binary mask. Tubular structures yield a small number of long
  components; noise yields many tiny components.
* **Skeleton length** — number of voxels in the 3D medial-axis skeleton
  (``skimage.morphology.skeletonize`` with method ``"lee"``). A scale-aware
  proxy for total vessel length in voxels.
* **Otsu threshold** — Otsu cutoff on the brain-masked soft response.
* **Jaccard / Dice** — set-overlap between two binary masks, restricted to
  the brain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


class VesselMaskAnalysisError(Exception):
    """Raised at the analysis boundary for shape / dtype contract violations."""


# ---------------------------------------------------------------------- helpers


def _check_pair(a: NDArray[Any], b: NDArray[Any], label: str) -> None:
    if a.shape != b.shape:
        raise VesselMaskAnalysisError(
            f"{label}: shape mismatch ({a.shape} vs {b.shape})"
        )


# ---------------------------------------------------------------- metric primitives


def binary_fraction(
    soft: NDArray[np.floating[Any]],
    brain: NDArray[Any],
    threshold: float,
) -> float:
    """Fraction of brain voxels with ``soft >= threshold``.

    Parameters
    ----------
    soft
        Soft response, any float dtype.
    brain
        Binary brain mask (``bool`` or ``{0,1}`` numeric).
    threshold
        Soft threshold in the same units as ``soft`` (expected ``[0, 1]``).

    Returns
    -------
    float
        Vessel-positive voxel fraction within the brain in ``[0, 1]``.
    """
    _check_pair(soft, brain, "binary_fraction")
    brain_b = brain.astype(bool, copy=False)
    n_brain = int(brain_b.sum())
    if n_brain == 0:
        return 0.0
    binary = (soft >= threshold) & brain_b
    return float(binary.sum()) / float(n_brain)


def connected_components_stats(
    binary: NDArray[Any],
    brain: NDArray[Any] | None = None,
) -> dict[str, Any]:
    """Return CC statistics for a 3D binary mask.

    Returns
    -------
    dict
        Keys:

        * ``n_components`` — total number of 26-connected components.
        * ``total_voxels`` — voxels in the binary mask (within brain if given).
        * ``largest_voxels`` — voxel count of the largest component.
        * ``median_voxels`` — median component size.
        * ``small_cc_count`` — components with size < 10 voxels (a noise proxy).
        * ``largest_fraction`` — ``largest_voxels / total_voxels``.
    """
    b = binary.astype(bool, copy=False)
    if brain is not None:
        _check_pair(b, brain, "connected_components_stats")
        b = b & brain.astype(bool, copy=False)
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labels, n = ndimage.label(b, structure=structure)
    total = int(b.sum())
    if n == 0:
        return {
            "n_components": 0,
            "total_voxels": 0,
            "largest_voxels": 0,
            "median_voxels": 0,
            "small_cc_count": 0,
            "largest_fraction": 0.0,
        }
    sizes = np.bincount(labels.ravel())[1:]  # skip background
    largest = int(sizes.max())
    return {
        "n_components": int(n),
        "total_voxels": total,
        "largest_voxels": largest,
        "median_voxels": int(np.median(sizes)),
        "small_cc_count": int((sizes < 10).sum()),
        "largest_fraction": float(largest) / float(total) if total > 0 else 0.0,
    }


def skeleton_length(binary: NDArray[Any], brain: NDArray[Any] | None = None) -> int:
    """Number of voxels in the 3D medial-axis skeleton of ``binary``.

    Uses :func:`skimage.morphology.skeletonize` (Lee 1994; 3D thinning). The
    value is a scale-aware proxy for total vessel length in voxels — long
    tubular structures contribute O(L) voxels; spherical blobs contribute O(1).
    """
    b = binary.astype(bool, copy=False)
    if brain is not None:
        _check_pair(b, brain, "skeleton_length")
        b = b & brain.astype(bool, copy=False)
    if not b.any():
        return 0
    skel = skeletonize(b, method="lee")
    return int(skel.astype(bool).sum())


def otsu_threshold_brainmasked(
    soft: NDArray[np.floating[Any]],
    brain: NDArray[Any],
    nbins: int = 256,
) -> float:
    """Otsu cutoff on the brain-restricted soft response."""
    _check_pair(soft, brain, "otsu_threshold_brainmasked")
    brain_b = brain.astype(bool, copy=False)
    values = soft[brain_b]
    if values.size == 0 or float(values.max()) <= float(values.min()):
        return 0.0
    return float(threshold_otsu(values, nbins=nbins))


def _intersect_brain(
    a: NDArray[Any], b: NDArray[Any], brain: NDArray[Any]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    _check_pair(a, b, "intersect_brain")
    _check_pair(a, brain, "intersect_brain")
    brain_b = brain.astype(bool, copy=False)
    a_b = a.astype(bool, copy=False) & brain_b
    b_b = b.astype(bool, copy=False) & brain_b
    return a_b, b_b


def jaccard(
    a: NDArray[Any], b: NDArray[Any], brain: NDArray[Any]
) -> float:
    """Jaccard index ``|A ∩ B| / |A ∪ B|`` over the brain mask."""
    a_b, b_b = _intersect_brain(a, b, brain)
    inter = int((a_b & b_b).sum())
    union = int((a_b | b_b).sum())
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def dice(
    a: NDArray[Any], b: NDArray[Any], brain: NDArray[Any]
) -> float:
    """Dice coefficient ``2 |A ∩ B| / (|A| + |B|)`` over the brain mask."""
    a_b, b_b = _intersect_brain(a, b, brain)
    n_a = int(a_b.sum())
    n_b = int(b_b.sum())
    if n_a + n_b == 0:
        return float("nan")
    inter = int((a_b & b_b).sum())
    return 2.0 * float(inter) / float(n_a + n_b)


# ---------------------------------------------------------------- sweep records


@dataclass(frozen=True)
class PerPatientSweepRecord:
    """All metrics for one ``(tag, patient, threshold)`` combination."""

    tag: str
    patient_id: str
    threshold: float
    binary_fraction: float
    n_components: int
    largest_fraction: float
    small_cc_count: int
    skeleton_voxels: int


@dataclass(frozen=True)
class PerTagSummary:
    """Aggregate metrics across all patients for one ``(tag, threshold)``."""

    tag: str
    threshold: float
    binary_fraction_mean: float
    binary_fraction_std: float
    binary_fraction_cv: float
    n_components_median: float
    largest_fraction_median: float
    small_cc_count_median: float
    skeleton_voxels_median: float
    n_patients: int


@dataclass(frozen=True)
class ThresholdSweepResult:
    """Bundle returned by :func:`sweep_thresholds`."""

    per_patient: tuple[PerPatientSweepRecord, ...]
    per_tag_summary: tuple[PerTagSummary, ...]
    otsu_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)


def sweep_thresholds(
    *,
    tag: str,
    patient_id: str,
    soft: NDArray[np.floating[Any]],
    brain: NDArray[Any],
    thresholds: tuple[float, ...] | list[float],
) -> list[PerPatientSweepRecord]:
    """Compute all per-threshold metrics for one (tag, patient).

    The Otsu cutoff is computed once on the brain-masked soft response and
    returned separately by the caller — including it in the sweep would mix
    a data-driven threshold with the fixed sweep grid.
    """
    out: list[PerPatientSweepRecord] = []
    brain_b = brain.astype(bool, copy=False)
    for t in thresholds:
        binary = (soft >= float(t)) & brain_b
        ccs = connected_components_stats(binary, brain_b)
        skel_v = skeleton_length(binary, brain_b)
        out.append(
            PerPatientSweepRecord(
                tag=tag,
                patient_id=patient_id,
                threshold=float(t),
                binary_fraction=binary_fraction(soft, brain_b, float(t)),
                n_components=ccs["n_components"],
                largest_fraction=ccs["largest_fraction"],
                small_cc_count=ccs["small_cc_count"],
                skeleton_voxels=skel_v,
            )
        )
    return out


def aggregate_per_tag(
    records: list[PerPatientSweepRecord],
    *,
    epsilon: float = 1e-8,
) -> list[PerTagSummary]:
    """Reduce per-patient sweep records to per-(tag, threshold) summaries."""
    by_key: dict[tuple[str, float], list[PerPatientSweepRecord]] = {}
    for r in records:
        by_key.setdefault((r.tag, r.threshold), []).append(r)
    out: list[PerTagSummary] = []
    for (tag, t), rs in sorted(by_key.items()):
        bf = np.asarray([r.binary_fraction for r in rs], dtype=np.float64)
        out.append(
            PerTagSummary(
                tag=tag,
                threshold=float(t),
                binary_fraction_mean=float(bf.mean()),
                binary_fraction_std=float(bf.std(ddof=0)),
                binary_fraction_cv=float(bf.std(ddof=0) / (bf.mean() + epsilon)),
                n_components_median=float(
                    np.median([r.n_components for r in rs])
                ),
                largest_fraction_median=float(
                    np.median([r.largest_fraction for r in rs])
                ),
                small_cc_count_median=float(
                    np.median([r.small_cc_count for r in rs])
                ),
                skeleton_voxels_median=float(
                    np.median([r.skeleton_voxels for r in rs])
                ),
                n_patients=len(rs),
            )
        )
    return out


def pick_threshold_by_anatomical_fraction(
    summaries: list[PerTagSummary],
    *,
    target_fraction_range: tuple[float, float],
) -> dict[str, dict[str, Any]]:
    """Choose, per tag, the threshold whose mean binary fraction is closest to
    the midpoint of ``target_fraction_range`` and lands inside the band.

    Falls back to "closest by distance to midpoint" when no threshold lands in
    the band, in which case the rationale field flags this explicitly.

    Returns
    -------
    dict
        ``{tag: {"threshold": float, "binary_fraction_mean": float,
        "binary_fraction_cv": float, "rationale": str, "in_band": bool}}``.
    """
    lo, hi = float(target_fraction_range[0]), float(target_fraction_range[1])
    if hi < lo:
        raise ValueError("target_fraction_range must be (lo, hi) with lo <= hi")
    midpoint = 0.5 * (lo + hi)

    by_tag: dict[str, list[PerTagSummary]] = {}
    for s in summaries:
        by_tag.setdefault(s.tag, []).append(s)

    out: dict[str, dict[str, Any]] = {}
    for tag, items in by_tag.items():
        in_band = [s for s in items if lo <= s.binary_fraction_mean <= hi]
        if in_band:
            best = min(
                in_band, key=lambda s: abs(s.binary_fraction_mean - midpoint)
            )
            rationale = (
                f"binary fraction {best.binary_fraction_mean:.3f} lies inside "
                f"the anatomical band [{lo:.2f}, {hi:.2f}]; closest to midpoint "
                f"{midpoint:.3f} among {len(in_band)} in-band thresholds"
            )
            band_flag = True
        else:
            best = min(items, key=lambda s: abs(s.binary_fraction_mean - midpoint))
            rationale = (
                "no swept threshold lands inside the anatomical band "
                f"[{lo:.2f}, {hi:.2f}]; picked closest by distance to midpoint "
                f"{midpoint:.3f}. EXTEND THE SWEEP."
            )
            band_flag = False
        out[tag] = {
            "threshold": float(best.threshold),
            "binary_fraction_mean": float(best.binary_fraction_mean),
            "binary_fraction_cv": float(best.binary_fraction_cv),
            "n_components_median": float(best.n_components_median),
            "skeleton_voxels_median": float(best.skeleton_voxels_median),
            "rationale": rationale,
            "in_band": bool(band_flag),
        }
    return out

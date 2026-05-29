"""Geometry helpers operating on volume arrays and affines."""

from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)
from numpy.typing import NDArray


def reorient_to_axcodes(
    array: NDArray[Any],
    affine: NDArray[np.floating[Any]],
    axcodes: tuple[str, str, str] = ("L", "P", "S"),
) -> NDArray[Any]:
    """Reorient a volume's voxel axes to a target anatomical orientation.

    All cohorts in the corpus must share one voxel orientation so a fixed crop
    box and the frozen MAISI VAE see consistent anatomy. UCSF-PDGM is natively
    ``LPS``; BraTS-GLI is ``LAS`` (anterior-posterior flipped). Both are mapped
    to ``LPS`` (the default) before cropping/encoding.

    Parameters
    ----------
    array
        Voxel data of shape ``(X, Y, Z)`` in the source orientation.
    affine
        4×4 voxel-to-world affine for ``array``.
    axcodes
        Target orientation axis codes (default ``("L", "P", "S")``).

    Returns
    -------
    NDArray
        ``array`` reoriented to ``axcodes`` (axis permutations/flips only; no
        resampling, so voxel spacing is preserved). For an array already in the
        target orientation this is the identity.
    """
    src = io_orientation(affine)
    dst = axcodes2ornt(axcodes)
    transform = ornt_transform(src, dst)
    return apply_orientation(np.asarray(array), transform)


def array_axcodes(affine: NDArray[np.floating[Any]]) -> tuple[str, str, str]:
    """Return the anatomical axis codes of an affine, e.g. ``("L", "P", "S")``."""
    return tuple(nib.aff2axcodes(affine))  # type: ignore[return-value]


def brain_z_extent(mask: NDArray[Any], axial_axis: int = 2) -> tuple[int, int]:
    """First and last index along ``axial_axis`` where ``mask`` has non-zero voxels.

    Parameters
    ----------
    mask
        Boolean or numeric array; any non-zero voxel counts as "inside the brain".
    axial_axis
        Axis index treated as the axial (Z) direction. Defaults to 2, the LPS
        convention used by UCSF-PDGM.

    Returns
    -------
    (z_min, z_max) : tuple[int, int]
        Inclusive bounds. If the mask is empty, returns ``(0, mask.shape[axial_axis] - 1)``
        as a safe fallback.
    """
    occupied = np.any(mask != 0, axis=tuple(i for i in range(mask.ndim) if i != axial_axis))
    nz = np.flatnonzero(occupied)
    if nz.size == 0:
        return 0, mask.shape[axial_axis] - 1
    return int(nz[0]), int(nz[-1])


def non_empty_indices(mask: NDArray[Any], axis: int = 2, min_voxels: int = 1) -> list[int]:
    """Indices along ``axis`` whose slice contains at least ``min_voxels`` non-zero voxels.

    Use this when you need to pick representative slices that are *guaranteed*
    to have content — `brain_z_extent` only returns the bounding interval and
    can include slices that are empty inside the bounds.

    Parameters
    ----------
    mask
        Boolean or numeric array. Any non-zero voxel is counted.
    axis
        Axis along which to count (default 2 = axial Z in LPS).
    min_voxels
        Minimum non-zero voxels per slice to qualify as "non-empty".

    Returns
    -------
    list[int]
        Sorted list of slice indices that satisfy the threshold.
    """
    other = tuple(i for i in range(mask.ndim) if i != axis)
    counts = np.count_nonzero(mask, axis=other)
    return np.flatnonzero(counts >= min_voxels).tolist()


def evenly_spaced_indices(low: int, high: int, n: int) -> list[int]:
    """``n`` integer indices evenly spaced in the inclusive range ``[low, high]``."""
    if n <= 0:
        return []
    if high <= low:
        return [low] * n
    return [round(float(x)) for x in np.linspace(low, high, n)]


def pick_evenly_from(indices: list[int], n: int) -> list[int]:
    """Pick ``n`` items evenly distributed over the *positions* of a sorted index list.

    Unlike :func:`evenly_spaced_indices`, this guarantees every returned value
    is a member of ``indices`` — so if ``indices`` lists non-empty slices, every
    pick is itself non-empty.
    """
    if n <= 0 or not indices:
        return []
    if len(indices) <= n:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, n).round().astype(int)
    return [indices[int(i)] for i in positions]

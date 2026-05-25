"""Synthetic phantoms for OOF / Frangi correctness tests.

Provides three primitives:

* :func:`cylinder_volume` — single straight cylinder along an axis.
* :func:`rotated_cylinder_volume` — cylinder oriented along an arbitrary unit
  vector. Used for the orientation-invariance test (OOF should give a
  rotation-invariant response; Frangi has a known 5-15 % anisotropy bias).
* :func:`parallel_cylinders_volume` — two parallel cylinders at fixed
  separation. Used for the adjacency test (OOF separates them down to
  ``d ≈ 2 r``; Frangi merges them — the deep-medullary-vein failure mode of
  Bériault 2015).

All phantoms are dark-on-bright by default (SWI convention; magnitude is low
in veins). Pass ``background < foreground`` for bright tubes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def smooth_cylinder_volume(
    size: int,
    radius_mm: float,
    axis: int = 2,
    background: float = 0.8,
    foreground: float = 0.1,
    spacing_mm: float = 1.0,
    transition_mm: float = 0.5,
) -> NDArray[np.float32]:
    """Anti-aliased cylinder with a smooth radial transition.

    Used for peak-radius diagnostics where the *analytical* cylinder geometry
    matters. A hard binary cylinder on a 1 mm grid has effective radius
    ``a + 0.5`` mm (voxel partial coverage shifts the gradient transition),
    which biases the OOF peak by half a voxel. The smooth-boundary cylinder
    replaces the binary step with a sigmoid of width ``transition_mm`` so the
    centre-of-gradient lies at ``radius_mm`` independently of voxel size.

    Parameters
    ----------
    transition_mm
        Half-width of the sigmoid transition zone. Defaults to half a voxel,
        which gives a sharp-but-not-aliased edge.
    """
    if size < 2:
        raise ValueError("size must be >= 2")
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if spacing_mm <= 0:
        raise ValueError("spacing_mm must be positive")
    if transition_mm <= 0:
        raise ValueError("transition_mm must be positive")

    grid = np.indices((size, size, size), dtype=np.float32)
    centre = (size - 1) / 2.0
    cross_axes = [a for a in (0, 1, 2) if a != axis]
    r = np.sqrt(
        (grid[cross_axes[0]] - centre) ** 2 + (grid[cross_axes[1]] - centre) ** 2
    ) * spacing_mm
    # tanh sigmoid centred at radius_mm: fg inside, bg outside, smooth on edge.
    inside_weight = 0.5 * (1.0 - np.tanh((r - radius_mm) / transition_mm))
    img = (
        foreground * inside_weight + background * (1.0 - inside_weight)
    ).astype(np.float32)
    return img


def cylinder_volume(
    size: int,
    radius_mm: float,
    axis: int = 2,
    background: float = 0.8,
    foreground: float = 0.1,
    spacing_mm: float = 1.0,
) -> NDArray[np.float32]:
    """Build a 3D cube with a straight cylinder along one of the principal axes.

    Parameters
    ----------
    size
        Cube side length in voxels (output shape is ``(size, size, size)``).
    radius_mm
        Cylinder radius in millimetres.
    axis
        Axis along which the cylinder runs (``0``, ``1`` or ``2``).
    background, foreground
        Intensity outside / inside the cylinder. ``background > foreground``
        yields a dark tube (SWI vein convention).
    spacing_mm
        Voxel spacing (isotropic).

    Returns
    -------
    NDArray[np.float32]
        ``(size, size, size)`` float32 volume.
    """
    if size < 2:
        raise ValueError("size must be >= 2")
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if spacing_mm <= 0:
        raise ValueError("spacing_mm must be positive")

    grid = np.indices((size, size, size), dtype=np.float32)
    centre = (size - 1) / 2.0
    cross_axes = [a for a in (0, 1, 2) if a != axis]
    r2 = (grid[cross_axes[0]] - centre) ** 2 + (grid[cross_axes[1]] - centre) ** 2
    r2_mm = r2 * (spacing_mm**2)
    img = np.full((size, size, size), background, dtype=np.float32)
    img[r2_mm < radius_mm**2] = foreground
    return img


def rotated_cylinder_volume(
    size: int,
    radius_mm: float,
    direction: tuple[float, float, float],
    background: float = 0.8,
    foreground: float = 0.1,
    spacing_mm: float = 1.0,
) -> NDArray[np.float32]:
    """Cylinder oriented along an arbitrary unit-length 3-vector.

    Parameters
    ----------
    size, radius_mm, background, foreground, spacing_mm
        See :func:`cylinder_volume`.
    direction
        3-vector pointing along the cylinder axis. Normalised internally.

    Returns
    -------
    NDArray[np.float32]
        ``(size, size, size)`` float32 volume.
    """
    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm == 0:
        raise ValueError("direction must be non-zero")
    d = d / norm

    grid = np.indices((size, size, size), dtype=np.float32)
    centre = (size - 1) / 2.0
    # Vector from centre to each voxel.
    v = np.stack([grid[i] - centre for i in range(3)], axis=0)  # (3, X, Y, Z)
    # Perpendicular distance² = |v|² − (v · d)²
    v_dot_d = (
        v[0] * np.float32(d[0]) + v[1] * np.float32(d[1]) + v[2] * np.float32(d[2])
    )
    v2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    perp2 = np.maximum(0.0, v2 - v_dot_d**2)
    perp2_mm = perp2 * (spacing_mm**2)
    img = np.full((size, size, size), background, dtype=np.float32)
    img[perp2_mm < radius_mm**2] = foreground
    return img


def parallel_cylinders_volume(
    size: int,
    radius_mm: float,
    separation_mm: float,
    axis: int = 2,
    offset_axis: int = 1,
    background: float = 0.8,
    foreground: float = 0.1,
    spacing_mm: float = 1.0,
) -> NDArray[np.float32]:
    """Two parallel cylinders at fixed centre-to-centre separation.

    Parameters
    ----------
    size, radius_mm, background, foreground, spacing_mm
        See :func:`cylinder_volume`.
    separation_mm
        Centre-to-centre distance between the two cylinders (mm).
    axis
        Cylinder long axis.
    offset_axis
        Axis along which the two cylinders are offset. Must differ from
        ``axis``.

    Returns
    -------
    NDArray[np.float32]
        ``(size, size, size)`` float32 volume.
    """
    if axis == offset_axis:
        raise ValueError("offset_axis must differ from axis")
    if separation_mm <= 0:
        raise ValueError("separation_mm must be positive")

    grid = np.indices((size, size, size), dtype=np.float32)
    centre = (size - 1) / 2.0
    half_sep_vox = separation_mm / (2.0 * spacing_mm)

    cross_axes = [a for a in (0, 1, 2) if a != axis]
    # Distance from each cylinder's centreline.
    other_axis = next(a for a in cross_axes if a != offset_axis)
    r2_a = (grid[offset_axis] - centre - half_sep_vox) ** 2 + (
        grid[other_axis] - centre
    ) ** 2
    r2_b = (grid[offset_axis] - centre + half_sep_vox) ** 2 + (
        grid[other_axis] - centre
    ) ** 2
    r2_a_mm = r2_a * (spacing_mm**2)
    r2_b_mm = r2_b * (spacing_mm**2)
    img = np.full((size, size, size), background, dtype=np.float32)
    img[(r2_a_mm < radius_mm**2) | (r2_b_mm < radius_mm**2)] = foreground
    return img

"""Signed distance transforms for tumour region masks.

Provides two operators:

- ``euclidean_percomponent`` — Euclidean SDT computed independently per
  connected component (via :func:`scipy.ndimage.label`), then unioned by
  element-wise max.  Ensures that the space *between* two disconnected lesions
  is not scored as interior (the correct behaviour for multifocal NETC).

- ``geodesic`` — intensity-weighted distance via minimum-cost shortest path
  (SiNGR-inspired, Nair et al. 2020).  Edge cost is proportional to local
  image intensity so high-intensity barriers increase the effective distance.
  Interior distances still use Euclidean EDT (lesion interiors carry no
  meaningful routing information).  Requires an image array.

Sign convention (both modes):
    SDT > 0  inside the region
    SDT = 0  at the boundary
    SDT < 0  outside the region
    Values clipped to ±clip_vox before return.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.ndimage import label as nd_label

from vena.segmentation.exceptions import SegTargetError

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["signed_distance"]

# Extra voxels added around each component's bounding box before its EDT is
# computed. One voxel guarantees the component is surrounded by background
# inside the sub-volume (so the interior EDT is exact); the clip radius
# guarantees every voxel whose true SDT exceeds -clip_vox lies inside the box.
# See `_euclidean_percomponent` for the exactness argument.
_BBOX_PAD_MARGIN_VOX: int = 1


def _edt_single_mask(mask: NDArray) -> NDArray:
    """Euclidean signed distance for a single binary mask.

    SDT > 0 inside, < 0 outside, 0 at the boundary.

    Parameters
    ----------
    mask : NDArray
        Boolean array.

    Returns
    -------
    NDArray
        Float32 SDT array of the same shape as *mask*.
    """
    d_in = distance_transform_edt(mask).astype(np.float32)
    d_out = distance_transform_edt(~mask).astype(np.float32)
    return d_in - d_out


def _padded_bbox(
    obj_slices: tuple[slice, ...],
    shape: tuple[int, ...],
    pad: int,
) -> tuple[slice, ...]:
    """Grow a ``find_objects`` bounding box by *pad* voxels, clamped to *shape*."""
    return tuple(
        slice(max(0, sl.start - pad), min(dim, sl.stop + pad))
        for sl, dim in zip(obj_slices, shape, strict=True)
    )


def _euclidean_percomponent(mask: NDArray, clip_vox: float | None = None) -> NDArray:
    """Euclidean SDT per connected component, unioned by element-wise max.

    For each 26-connected component of *mask* the SDT is computed
    independently.  The union (element-wise max) ensures that the space
    *between* two disconnected lesions is never scored as interior by one
    component "reaching across" to the other.

    When *clip_vox* is supplied each component's EDT is evaluated on its own
    bounding box grown by ``ceil(clip_vox) + 1`` voxels instead of on the full
    volume.  **The clipped result is identical to the full-volume computation**:

    * *Interior* — the padded box contains the whole component surrounded by at
      least one background voxel, so ``distance_transform_edt`` inside the box
      measures distance to the component's own boundary, as it would globally.
    * *Exterior, inside the box* — the nearest component voxel is by
      construction inside the box, so the distance is the true one.
    * *Exterior, outside the box* — such a voxel is at least
      ``ceil(clip_vox) + 1 > clip_vox`` voxels away from every component voxel,
      so its true SDT is below ``-clip_vox`` and clips to exactly the
      ``-clip_vox`` value the accumulator is pre-filled with.

    Cost drops from ``2 · n_components`` full-volume Euclidean transforms to
    ``2 · n_components`` transforms over lesion-sized sub-volumes.  This matters:
    on a native ``(240, 240, 155)`` UCSF-PDGM label the full-volume form costs
    13–41 s per scan, which the training dataset would otherwise pay **per
    sample, per epoch**.

    Parameters
    ----------
    mask : NDArray
        3-D boolean array.
    clip_vox : float or None
        Clip radius the caller will apply.  ``None`` reproduces the original
        full-volume computation exactly (unclipped, ``-inf`` far field) and is
        retained so the function stays correct for callers that do not clip.

    Returns
    -------
    NDArray
        Float32 SDT array, same shape as *mask*.  Positive inside each
        component, negative outside, zero at each component's boundary.
    """
    far_field = np.float32(-np.inf) if clip_vox is None else np.float32(-clip_vox)

    if not mask.any():
        # No foreground — every voxel is maximally outside.
        return np.full(mask.shape, far_field, dtype=np.float32)

    # 26-connectivity so corner-touching voxels form a single component.
    struct = np.ones((3, 3, 3), dtype=bool)
    labelled, n_comps = nd_label(mask, structure=struct)

    if clip_vox is None:
        # Exact unclipped path — must evaluate each EDT over the full volume,
        # because the far field carries real (unbounded) distances.
        if n_comps == 1:
            return _edt_single_mask(mask)
        union_sdt = np.full(mask.shape, far_field, dtype=np.float32)
        for comp_idx in range(1, n_comps + 1):
            comp_sdt = _edt_single_mask(labelled == comp_idx)
            np.maximum(union_sdt, comp_sdt, out=union_sdt)
        return union_sdt

    pad = int(np.ceil(clip_vox)) + _BBOX_PAD_MARGIN_VOX
    union_sdt = np.full(mask.shape, far_field, dtype=np.float32)

    # find_objects returns one bounding box per label in a single pass, so the
    # full volume is never scanned once per component.
    for comp_idx, obj_slices in enumerate(find_objects(labelled), start=1):
        if obj_slices is None:  # label index absent (cannot happen post-nd_label)
            continue
        box = _padded_bbox(obj_slices, mask.shape, pad)
        sub_sdt = _edt_single_mask(labelled[box] == comp_idx)
        # Basic slicing yields a view, so this writes back into union_sdt.
        np.maximum(union_sdt[box], sub_sdt, out=union_sdt[box])

    return union_sdt


def _geodesic(mask: NDArray, image: NDArray) -> NDArray:
    """Geodesic SDT via image-intensity-weighted shortest path.

    Interior distances use Euclidean EDT (within-lesion routing has no
    clinical meaning).  Exterior distances are minimum-cost paths on a
    fully-discretised 6-connected 3-D grid whose edge costs are proportional
    to the average normalised intensity of the two endpoint voxels.  High-
    intensity barriers therefore increase the effective distance to the mask.

    Parameters
    ----------
    mask : NDArray
        3-D boolean array.
    image : NDArray
        3-D float array of the same shape as *mask*.  Raw or pre-normalised
        intensities — higher values increase traversal cost.

    Returns
    -------
    NDArray
        Float32 SDT array, same shape as *mask*.  Positive inside (Euclidean),
        negative outside (cost-weighted).  Scales differ across the boundary
        when barriers are present; the 0.5-sigmoid contour still lies at the
        mask boundary within one voxel.

    Raises
    ------
    SegTargetError
        If *image* shape does not match *mask* shape, or if *mask* has no
        foreground (no seeds for the shortest-path solver).
    """
    from skimage.graph import MCP_Geometric  # optional heavy import — only on geodesic path

    if image.shape != mask.shape:
        raise SegTargetError(f"image shape {image.shape} does not match mask shape {mask.shape}")
    if not mask.any():
        raise SegTargetError("geodesic SDT requires at least one foreground voxel (no MCP seeds)")

    _eps = 1e-8
    img_max = float(image.max())
    image_norm = image.astype(np.float64) / (img_max + _eps)
    # Cost per voxel: baseline 1 + normalised intensity (range [1, 2])
    cost = (1.0 + image_norm).astype(np.float64)

    # Euclidean distance from interior voxels to mask boundary (positive inside)
    d_in = distance_transform_edt(mask).astype(np.float32)

    # Minimum-cost path from any mask voxel to each exterior point (negative outside)
    seeds = np.argwhere(mask)  # shape (n_seeds, ndim)
    mcp = MCP_Geometric(cost, fully_connected=False)
    geo_dist_to_mask, _ = mcp.find_costs(seeds)

    # Combine: inside → Euclidean; outside → negative geodesic cost
    sdt = np.where(mask, d_in, -geo_dist_to_mask.astype(np.float32))
    return sdt.astype(np.float32)


def signed_distance(
    mask: NDArray,
    *,
    mode: Literal["euclidean_percomponent", "geodesic"],
    image: NDArray | None = None,
    clip_vox: float,
) -> NDArray:
    """Compute the signed distance transform of *mask*.

    Parameters
    ----------
    mask : NDArray
        Boolean array of arbitrary shape (typically 3-D ``(H, W, D)``).
    mode : {"euclidean_percomponent", "geodesic"}
        SDT operator.  ``"euclidean_percomponent"`` applies per-connected-
        component Euclidean SDT unioned by max — correct for multifocal
        lesions.  ``"geodesic"`` computes intensity-weighted shortest-path
        distances and requires *image*.
    image : NDArray or None
        Intensity array, same shape as *mask*.  Required when
        ``mode="geodesic"``; silently unused otherwise.
    clip_vox : float
        Absolute SDT values are clipped to ``[-clip_vox, +clip_vox]`` before
        returning.  Prevents extreme gradient magnitudes from large background
        regions.

    Returns
    -------
    NDArray
        Float32 array of the same shape as *mask*.  SDT > 0 inside, < 0
        outside, ≈ 0 at the boundary, clipped to ``±clip_vox``.

    Raises
    ------
    SegTargetError
        If ``mode="geodesic"`` but *image* is ``None``, if *image* shape
        mismatches *mask*, or if an unknown mode is supplied.
    """
    if mode == "geodesic" and image is None:
        raise SegTargetError("mode='geodesic' requires image to be provided")

    mask_bool = np.asarray(mask, dtype=bool)

    if mode == "euclidean_percomponent":
        # clip_vox is forwarded so each component's EDT runs on its own padded
        # bounding box; the clipped result is identical to the full-volume form.
        sdt = _euclidean_percomponent(mask_bool, clip_vox=clip_vox)
    elif mode == "geodesic":
        sdt = _geodesic(mask_bool, np.asarray(image, dtype=np.float32))
    else:
        raise SegTargetError(f"Unknown SDT mode: {mode!r}")

    return np.clip(sdt, -clip_vox, clip_vox).astype(np.float32)

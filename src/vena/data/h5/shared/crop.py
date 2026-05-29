"""Brain-centred crop-origin computation (data layer, numpy-only).

The multi-cohort corpus stores each cohort's native-shape volume plus a
per-scan crop origin; the MAISI encoder later crops/pads every cohort onto a
common box so latents share a spatial grid. This module owns the *geometry*
computed at conversion time (where the box starts in the native grid). The
tensor crop/pad ops that consume it live in the model layer
(``vena.model.autoencoder.maisi.preprocessing``); keeping the origin
computation here avoids a data→model import.

The box is defined in canonical-LPS voxel space (axes L→R, P→A, S→I), matching
the ``(H, W, D)`` order of the stored arrays. All cohorts are reoriented to LPS
before this is computed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Common crop/pad box (H, W, D) = (L-R, P-A, S-I), 1 mm iso. Chosen as the
#: unique minimal box that is ÷32 in image space (÷8 in latent space after the
#: 4× MAISI VAE) and contains every measured glioma brain with ≥~15 mm/side
#: margin. See the project plan for the derivation.
DEFAULT_CROP_BOX: tuple[int, int, int] = (192, 224, 192)


class CropGeometryError(ValueError):
    """Raised when the common box cannot contain a case's brain extent."""


def compute_crop_origin(
    brain_mask: NDArray[np.bool_] | NDArray[np.integer],
    target_box: tuple[int, int, int] = DEFAULT_CROP_BOX,
) -> tuple[int, int, int]:
    """Compute the brain-centred crop origin for a fixed box.

    The box is centred on the brain bounding-box centre. On axes where the
    native dimension is at least the box size the origin is clamped to keep the
    box inside the array (real data preferred to zero-padding); on smaller axes
    the origin stays centred and may be negative (→ the encoder zero-pads).
    Containment of the whole brain bounding box is asserted afterwards.

    Parameters
    ----------
    brain_mask : NDArray
        Binary brain region of shape ``(H, W, D)`` in canonical LPS.
    target_box : tuple[int, int, int]
        Box shape per axis ``(H, W, D)``.

    Returns
    -------
    tuple[int, int, int]
        ``crop_origin`` per axis (may be negative on padding axes).

    Raises
    ------
    ValueError
        If ``brain_mask`` is not 3-D.
    CropGeometryError
        If the box (centred and clamped) does not contain the brain bounding
        box on some axis — i.e. the box is smaller than the brain extent.
    """
    mask = np.asarray(brain_mask)
    if mask.ndim != 3:
        raise ValueError(
            f"compute_crop_origin expects a 3-D (H,W,D) mask; got {mask.shape}"
        )
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        # Empty mask: centre the box on the geometric centre of the array.
        return tuple(
            int(round(n / 2.0 - t / 2.0)) for n, t in zip(mask.shape, target_box)
        )  # type: ignore[return-value]

    lo = nz.min(axis=0)
    hi = nz.max(axis=0)  # inclusive
    origin: list[int] = []
    for i in range(3):
        centre = (int(lo[i]) + int(hi[i]) + 1) / 2.0
        o = int(round(centre - target_box[i] / 2.0))
        n = int(mask.shape[i])
        if n >= target_box[i]:
            o = int(np.clip(o, 0, n - target_box[i]))
        origin.append(o)

    for i in range(3):
        if not (origin[i] <= int(lo[i]) and int(hi[i]) < origin[i] + target_box[i]):
            raise CropGeometryError(
                f"crop box does not contain the brain on axis {i}: "
                f"bbox=({int(lo[i])},{int(hi[i])}), origin={origin[i]}, box={target_box[i]}. "
                "The common box is smaller than this case's brain extent."
            )
    return tuple(origin)  # type: ignore[return-value]

"""Small-angle 3D rotations around the LPS yaw (S) and roll (P) axes.

Implemented via ``F.affine_grid + F.grid_sample`` with normalised
``[-1, 1]`` coordinates so the SAME 4x4 affine matrix can be applied to both
image- and latent-grid tensors without any scale-dependent recomputation. The
rotation is therefore representation-agnostic: equivariance is a property of
the MAISI VAE, not of the operator itself.

Two concrete subclasses are exposed:

- :class:`RotateYaw` rotates around the LPS **S** axis (axial yaw — patient's
  head turning left/right). Affects the (H, W) plane; D-slices stay axial.
- :class:`RotateRoll` rotates around the LPS **P** axis (sagittal roll —
  patient's head tilting onto a shoulder). Affects the (H, D) plane.

A pitch rotation (around L) is intentionally omitted — clinically rare and
strongly limited by the brain's anisotropic field-of-view.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F

from vena.data.augment.base import LatentAugmentation, LatentAugmentationError


def _rotation_matrix(plane: str, angle_rad: float, device, dtype) -> torch.Tensor:
    """Build the 3x4 affine matrix expected by ``F.affine_grid``.

    The matrix is constructed in normalised ``[-1, 1]`` coordinates. ``plane``
    selects which two of the three axes rotate:

    - ``"yaw"`` — rotation around the S axis; permutes the (H, W) plane.
    - ``"roll"`` — rotation around the P axis; permutes the (H, D) plane.

    Returns a ``(3, 4)`` tensor.
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    # Identity affine in (z, y, x) order = (D, W, H) per F.affine_grid's
    # convention. Build the in-plane rotation, leave the third axis pure.
    if plane == "yaw":
        # Rotation around D (S axis); affects H ↔ W. F.affine_grid uses
        # (D, W, H) coordinate order, so the H-row is index 2 and the W-row
        # is index 1.
        mat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
    elif plane == "roll":
        # Rotation around W (P axis); affects H ↔ D. H row is index 2,
        # D row is index 0.
        mat = torch.tensor(
            [
                [c, 0.0, -s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [s, 0.0, c, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
    else:  # pragma: no cover — guarded at construction
        raise ValueError(f"_rotation_matrix: unknown plane {plane!r}")
    return mat


def _affine_resample_3d(
    vol: torch.Tensor, mat: torch.Tensor, padding_mode: str = "zeros"
) -> torch.Tensor:
    """Apply a 3x4 affine to a 3-D volume via ``F.grid_sample``.

    ``vol`` is ``(H, W, D)`` or ``(C, H, W, D)``; the function preserves the
    leading channel dim and returns the same shape.
    """
    has_channel = vol.ndim == 4
    if not has_channel:
        if vol.ndim != 3:
            raise ValueError(f"_affine_resample_3d: expected 3-D or 4-D; got {tuple(vol.shape)}")
        vol = vol.unsqueeze(0)
    x = vol.unsqueeze(0).float()  # (1, C, H, W, D)
    grid = F.affine_grid(mat.unsqueeze(0).to(x.dtype), size=x.shape, align_corners=False)
    y = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
    y = y[0]
    if not has_channel:
        y = y[0]
    return y.contiguous().to(vol.dtype)


class _RotateBase(LatentAugmentation):
    """Shared logic for plane-restricted small-angle rotations."""

    PLANE: str = ""  # subclass override: "yaw" or "roll"

    def __init__(self, p: float = 0.3, max_deg: float = 5.0) -> None:
        super().__init__(p=p)
        if float(max_deg) <= 0.0:
            raise LatentAugmentationError(f"{self.name}: max_deg must be positive; got {max_deg}")
        self.max_deg = float(max_deg)

    def sample_params(self, rng: random.Random) -> dict[str, Any]:
        # Continuous draw in [-max_deg, +max_deg]; clamped at boundaries.
        deg = rng.uniform(-self.max_deg, self.max_deg)
        return {"deg": float(deg)}

    def _make_matrix(self, deg: float, device, dtype) -> torch.Tensor:
        return _rotation_matrix(self.PLANE, math.radians(deg), device, dtype)

    def apply_latent(self, batch: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        deg = float(params["deg"])
        if deg == 0.0:
            return batch
        for key in self.LATENT_KEYS:
            if key in batch and isinstance(batch[key], torch.Tensor):
                t = batch[key]
                mat = self._make_matrix(deg, t.device, t.dtype)
                batch[key] = _affine_resample_3d(t, mat)
        return batch

    def apply_image(self, x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"{self.name}.apply_image expects (H,W,D); got shape {tuple(x.shape)}")
        deg = float(params["deg"])
        if deg == 0.0:
            return x
        mat = self._make_matrix(deg, x.device, x.dtype)
        return _affine_resample_3d(x, mat)

    def param_tag(self, params: dict[str, Any]) -> str:
        # Bucket the continuous angle into integer degrees so the combination
        # CSV does not explode into millions of unique tags. Sign is encoded
        # with ``p``/``n`` so the tag is safe inside the pipeline's
        # ``+``-separated combination string.
        deg_int = int(round(float(params["deg"])))
        sign = "p" if deg_int >= 0 else "n"
        return f"{self.name}_{sign}{abs(deg_int)}"


class RotateYaw(_RotateBase):
    """Axial yaw — rotation around the LPS S axis (H ↔ W plane)."""

    name = "rotate_yaw"
    PLANE = "yaw"


class RotateRoll(_RotateBase):
    """Sagittal roll — rotation around the LPS P axis (H ↔ D plane)."""

    name = "rotate_roll"
    PLANE = "roll"

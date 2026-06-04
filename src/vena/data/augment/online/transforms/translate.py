"""Integer-voxel translation, with the latent shift scaled by 4× compression."""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F

from vena.data.augment.online.base import LatentAugmentation, LatentAugmentationError

# MAISI-V2 spatial compression factor: 4× per axis. An integer-voxel shift of
# ``k`` voxels in image space corresponds to a shift of ``k // 4`` voxels in
# latent space; restricting image shifts to multiples of 4 keeps the two ops
# exactly aligned with no sub-pixel resampling.
_SPATIAL_COMPRESSION: int = 4


class Translate(LatentAugmentation):
    """Independent integer-voxel translations along each spatial axis.

    The shift on each axis is drawn uniformly from ``{-max_voxels, …, +max_voxels}``
    restricted to multiples of ``_SPATIAL_COMPRESSION`` (4). Latent-space shifts
    are exactly ``shift // 4`` voxels so the image- and latent-space operators
    are commutative under decode.

    Parameters
    ----------
    p : float
        Per-sample probability.
    max_voxels : int
        Maximum absolute shift in image-space voxels per axis. Must be a
        non-negative multiple of 4.
    axes : tuple[str, ...]
        Subset of ``{"h", "w", "d"}`` over which shifts are applied. Defaults
        to all three.
    """

    name = "translate"

    def __init__(
        self,
        p: float = 0.5,
        max_voxels: int = 8,
        axes: tuple[str, ...] = ("h", "w", "d"),
    ) -> None:
        super().__init__(p=p)
        if int(max_voxels) < 0 or int(max_voxels) % _SPATIAL_COMPRESSION != 0:
            raise LatentAugmentationError(
                f"Translate: max_voxels must be a non-negative multiple of "
                f"{_SPATIAL_COMPRESSION}; got {max_voxels}"
            )
        for a in axes:
            if a not in ("h", "w", "d"):
                raise LatentAugmentationError(
                    f"Translate: axes entries must be in {{h, w, d}}; got {a!r}"
                )
        self.max_voxels = int(max_voxels)
        self.axes: tuple[str, ...] = tuple(axes)

    # ------------------------------------------------------------------

    def sample_params(self, rng: random.Random) -> dict[str, Any]:
        steps_per_axis = self.max_voxels // _SPATIAL_COMPRESSION
        shifts: dict[str, int] = {"h": 0, "w": 0, "d": 0}
        for a in self.axes:
            # Draw an integer step in [-steps_per_axis, steps_per_axis] then
            # multiply by the compression factor → image-space shift.
            k = rng.randint(-steps_per_axis, steps_per_axis)
            shifts[a] = int(k) * _SPATIAL_COMPRESSION
        return {"shifts_img": shifts}

    # ------------------------------------------------------------------

    @staticmethod
    def _shift_volume(vol: torch.Tensor, shifts: tuple[int, int, int]) -> torch.Tensor:
        """Zero-padded integer shift on the last three axes of ``vol``.

        ``vol`` may be ``(H, W, D)`` or ``(C, H, W, D)`` — the shifts are
        applied to the trailing three spatial axes in both cases.
        """
        sh, sw, sd = shifts
        if sh == 0 and sw == 0 and sd == 0:
            return vol
        # F.pad operates on the last axes first. For (.., H, W, D) the order
        # is (D_left, D_right, W_left, W_right, H_left, H_right).
        pad = (
            max(0, sd),
            max(0, -sd),
            max(0, sw),
            max(0, -sw),
            max(0, sh),
            max(0, -sh),
        )
        padded = F.pad(vol, pad, mode="constant", value=0.0)
        # After padding, slice back to the original shape. The slice start is
        # ``max(0, -shift)``; equivalently, we drop the padding inserted on
        # the opposite side.
        h0 = max(0, -sh)
        w0 = max(0, -sw)
        d0 = max(0, -sd)
        H = vol.shape[-3]
        W = vol.shape[-2]
        D = vol.shape[-1]
        return padded[..., h0 : h0 + H, w0 : w0 + W, d0 : d0 + D].contiguous()

    # ------------------------------------------------------------------

    def apply_latent(self, batch: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        shifts_img = params["shifts_img"]
        latent_shifts = (
            shifts_img["h"] // _SPATIAL_COMPRESSION,
            shifts_img["w"] // _SPATIAL_COMPRESSION,
            shifts_img["d"] // _SPATIAL_COMPRESSION,
        )
        for key in self.LATENT_KEYS:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = self._shift_volume(batch[key], latent_shifts)
        return batch

    def apply_image(self, x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Translate.apply_image expects (H,W,D); got shape {tuple(x.shape)}")
        shifts_img = params["shifts_img"]
        return self._shift_volume(x, (shifts_img["h"], shifts_img["w"], shifts_img["d"]))

    def param_tag(self, params: dict[str, Any]) -> str:
        s = params["shifts_img"]

        # Zero-shift axes are dropped so the tag stays short. Sign is encoded
        # with ``p``/``n`` rather than ``+``/``-`` so the tag is safe to use
        # inside the pipeline's ``+``-separated combination string.
        def _enc(a: str, k: int) -> str:
            sign = "p" if k >= 0 else "n"
            return f"{a}{sign}{abs(k)}"

        parts = [_enc(a, s[a]) for a in ("h", "w", "d") if s[a] != 0]
        if not parts:
            return f"{self.name}_id"
        return self.name + "_" + "".join(parts)

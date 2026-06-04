"""TorchIO ↔ MONAI adapter shims for the offline augmentation bank.

TorchIO covers physics-grounded MR transforms (bias field, motion,
ghosting, anisotropy) but does not implement the piecewise-linear
monotonic intensity remap from Augment-to-Augment (Zimmermann 2025,
arXiv:2511.09366) and other MR-synthesis literature. MONAI's
:class:`monai.transforms.RandHistogramShift` does. This module wraps it
as a TorchIO :class:`~torchio.transforms.IntensityTransform` so the rest
of the variant builder can compose it alongside native TorchIO
operators.

Three implementation traps the Plan agent flagged are handled here:

1. **Channel-axis agreement** — both libraries are channel-first; no
   transpose is needed at the boundary.
2. **Intensity range** — MONAI's histogram-shift remaps along the
   ``[image.min(), image.max()]`` range; the input intensity is therefore
   not normalised. We pass the raw image through and let MONAI honour the
   per-image range it sees.
3. **Random state** — TorchIO seeds its own RNG per call. We re-seed
   MONAI's transform with an integer drawn from TorchIO's per-call random
   state so the shim is deterministic when TorchIO is.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torchio as tio
from monai.transforms import RandHistogramShift  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class MonaiHistogramShift(tio.IntensityTransform):  # type: ignore[misc]
    """Apply MONAI's piecewise-linear monotonic intensity remap to a Subject.

    Each included :class:`~torchio.ScalarImage` is processed independently:
    MONAI samples a sorted set of control points uniformly in
    ``[0, 1]`` (after the transform's own min-max normalisation) and
    interpolates the input intensities through the resulting piecewise-
    linear transfer function. The transform is monotonic by construction,
    so it never reverses brightness ordering.

    Parameters
    ----------
    num_control_points : tuple[int, int]
        Inclusive range from which the number of control points is drawn
        uniformly per call. Defaults to ``(8, 12)`` — Augment-to-Augment
        uses 4–8; we lean denser because our `[0, 1]` MR distribution is
        more multi-modal than the published study's range.
    prob : float
        Probability of firing per call. Defaults to ``1.0`` because the
        variant builder already gates each variant via TorchIO's
        :class:`OneOf` weights — passing ``prob<1`` here would double-gate.
    include : list[str] | None
        Forwarded to :class:`torchio.IntensityTransform`. When set, only
        the listed Subject keys are remapped (input-only contract).
    """

    def __init__(
        self,
        num_control_points: tuple[int, int] = (8, 12),
        prob: float = 1.0,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(include=include, **kwargs)
        if not (isinstance(num_control_points, tuple) and len(num_control_points) == 2):
            raise ValueError(
                f"num_control_points must be a (low, high) tuple; got {num_control_points!r}"
            )
        low, high = num_control_points
        if low < 3 or high < low:
            raise ValueError(
                f"num_control_points: need 3 <= low <= high; got {num_control_points!r}"
            )
        self.num_control_points = (int(low), int(high))
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1]; got {prob!r}")
        self.prob = float(prob)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply the histogram shift to every included ScalarImage of ``subject``."""
        for name, image in self.get_images_dict(subject).items():
            if not isinstance(image, tio.ScalarImage):
                continue
            data = image.data  # (C, H, W, D), float
            seed = int(torch.randint(0, np.iinfo(np.int32).max, (1,)).item())
            transform = RandHistogramShift(
                num_control_points=self.num_control_points,
                prob=self.prob,
            )
            transform.set_random_state(seed=seed)
            shifted_np = transform(np.asarray(data, dtype=np.float32))
            image.set_data(torch.from_numpy(np.asarray(shifted_np)))
        return subject

    def get_params_for_logging(self) -> dict[str, Any]:
        """Snapshot of sampled hyperparameters for the per-row provenance JSON."""
        return {
            "transform": "monai.RandHistogramShift",
            "num_control_points_range": list(self.num_control_points),
            "prob": self.prob,
        }

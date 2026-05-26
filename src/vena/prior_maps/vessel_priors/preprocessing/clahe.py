"""CLAHE — Contrast-Limited Adaptive Histogram Equalisation.

Wraps :func:`skimage.exposure.equalize_adapthist` for 3D SWI volumes,
restricted to the brain mask so skull / background voxels do not influence the
local histograms.

References
----------
Zuiderveld, K. *Contrast Limited Adaptive Histogram Equalization*. Graphics
Gems IV, 1994, pp. 474-485.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from skimage.exposure import equalize_adapthist

from vena.data.niigz import NiftiVolume
from vena.prior_maps.vessel_priors.abc_preprocessor import AbstractPreprocessor

logger = logging.getLogger(__name__)


class CLAHEPreprocessor(AbstractPreprocessor):
    """Brain-masked CLAHE on a 3D SWI volume.

    Parameters
    ----------
    clip_limit
        Contrast clip limit passed to :func:`equalize_adapthist`. Typical range
        ``[0.001, 0.05]``; lower values produce more conservative contrast
        enhancement. Default ``0.01``.
    nbins
        Histogram bin count. Default 256.
    kernel_size
        Optional kernel shape passed to :func:`equalize_adapthist`. When None,
        scikit-image uses ``input_shape / 8`` per axis. Accepts an int (uniform)
        or a 3-tuple ``(kx, ky, kz)``.
    percentile_clip
        Percentile range used to map the brain-masked SWI to ``[0, 1]`` before
        CLAHE. Default ``(0.5, 99.5)`` matches the Frangi normalisation step.
    """

    name: ClassVar[str] = "clahe"

    def __init__(
        self,
        clip_limit: float = 0.01,
        nbins: int = 256,
        kernel_size: int | tuple[int, int, int] | None = None,
        percentile_clip: tuple[float, float] = (0.5, 99.5),
    ) -> None:
        if not (0.0 < clip_limit <= 1.0):
            raise ValueError("clip_limit must lie in (0, 1]")
        if nbins < 2:
            raise ValueError("nbins must be >= 2")
        self.clip_limit = float(clip_limit)
        self.nbins = int(nbins)
        self.kernel_size = kernel_size
        self.percentile_clip = (float(percentile_clip[0]), float(percentile_clip[1]))

    def apply(self, volume: NiftiVolume, brain_mask: NDArray[Any]) -> NiftiVolume:
        arr = volume.array.astype(np.float32, copy=False)
        brain = (brain_mask > 0).astype(bool)
        if brain.shape != arr.shape:
            raise ValueError(f"brain mask shape {brain.shape} != volume shape {arr.shape}")

        # Percentile-stretch within the brain so CLAHE sees a well-bounded input
        # in [0, 1]. Voxels outside the brain are forced to 0 to keep them out
        # of the local histograms.
        in_brain = arr[brain]
        if in_brain.size == 0:
            logger.warning(
                "CLAHE: empty brain mask for %s, returning input unchanged",
                volume.path,
            )
            return volume
        lo = float(np.percentile(in_brain, self.percentile_clip[0]))
        hi = float(np.percentile(in_brain, self.percentile_clip[1]))
        if hi <= lo:
            logger.warning("CLAHE: degenerate intensity range, returning zeros")
            zeroed = np.zeros_like(arr, dtype=np.float32)
            return NiftiVolume(
                array=zeroed,
                affine=volume.affine,
                header=volume.header,
                path=volume.path,
                spacing_mm=volume.spacing_mm,
            )

        normalised = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        normalised = normalised * brain.astype(np.float32)

        equalised = equalize_adapthist(
            normalised,
            kernel_size=self.kernel_size,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        ).astype(np.float32)

        # Re-mask to suppress the contrast CLAHE introduces outside the brain.
        equalised = equalised * brain.astype(np.float32)

        return NiftiVolume(
            array=equalised,
            affine=volume.affine,
            header=volume.header,
            path=volume.path,
            spacing_mm=volume.spacing_mm,
        )

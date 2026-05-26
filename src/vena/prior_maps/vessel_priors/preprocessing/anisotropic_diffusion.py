"""Vessel-Enhancing Diffusion (VED) preprocessor via ITK-TubeTK.

Wraps :func:`itk.EnhanceTubesUsingDiffusion` for 3D SWI / SWAN volumes. ITK-
TubeTK ships Manniesing's vessel-enhancing diffusion (an anisotropic, Hessian-
steered diffusion that smooths along the vessel direction while suppressing
across-vessel noise), which is the spiritual successor of gradient anisotropic
diffusion for cerebrovascular imaging.

This preprocessor is intentionally *not* part of the default OOF pipeline (see
``routines/vessel_priors/configs/oof.yaml``) — Law & Chung 2008 showed the OOF
operator is already robust to multiplicative noise of the form found on SWI —
but is exposed so it can be enabled by a single YAML entry when a noisier
cohort warrants it.

References
----------
Manniesing, R., Viergever, M. A., Niessen, W. J. *Vessel enhancing diffusion:
A scale space representation of vessel structures*. Medical Image Analysis
10(6), 2006. https://doi.org/10.1016/j.media.2006.06.002

Aylward, S., Bullitt, E. *Initialization, noise, singularities, and scale in
height ridge traversal for tubular object centerline extraction*. IEEE TMI
21(2), 2002. https://doi.org/10.1109/42.993126
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import itk
import numpy as np
from numpy.typing import NDArray

from vena.data.niigz import NiftiVolume
from vena.prior_maps.vessel_priors.abc_preprocessor import (
    AbstractPreprocessor,
    PreprocessingError,
)

logger = logging.getLogger(__name__)


class AnisotropicDiffusionPreprocessor(AbstractPreprocessor):
    """Hessian-steered anisotropic diffusion (VED) on a 3D SWI volume.

    Parameters
    ----------
    n_iterations
        Number of diffusion iterations. Default ``5``.
    n_scales
        Number of Hessian scales scanned by the inner enhancement step.
        Default ``5``.
    sigma_min, sigma_max
        Inner Hessian σ range, in voxels. Default ``(0.5, 2.5)`` mirroring
        the OOF radius range on UCSF-PDGM (1 mm isotropic).
    time_step
        Diffusion time-step. Default ``0.01``.
    percentile_clip
        Percentile range used to map the brain-masked input to ``[0, 1]``
        before diffusion, mirroring the simple normalisation used elsewhere in
        the pipeline (no histogram-domain manipulation). Default
        ``(0.5, 99.5)``.
    """

    name: ClassVar[str] = "anisotropic_diffusion"

    def __init__(
        self,
        n_iterations: int = 5,
        n_scales: int = 5,
        sigma_min: float = 0.5,
        sigma_max: float = 2.5,
        time_step: float = 0.01,
        percentile_clip: tuple[float, float] = (0.5, 99.5),
    ) -> None:
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if n_scales < 1:
            raise ValueError("n_scales must be >= 1")
        if sigma_min <= 0 or sigma_max <= 0:
            raise ValueError("σ values must be positive")
        if sigma_max < sigma_min:
            raise ValueError("sigma_max must be >= sigma_min")
        if not (0.0 < time_step <= 1.0):
            raise ValueError("time_step must lie in (0, 1]")
        self.n_iterations = int(n_iterations)
        self.n_scales = int(n_scales)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.time_step = float(time_step)
        self.percentile_clip = (float(percentile_clip[0]), float(percentile_clip[1]))

    def apply(self, volume: NiftiVolume, brain_mask: NDArray[Any]) -> NiftiVolume:
        arr = volume.array.astype(np.float32, copy=False)
        brain = (brain_mask > 0).astype(bool)
        if brain.shape != arr.shape:
            raise PreprocessingError(f"brain mask shape {brain.shape} != volume shape {arr.shape}")

        # Simple percentile normalisation — no CLAHE / histogram manipulation.
        in_brain = arr[brain]
        if in_brain.size == 0:
            logger.warning(
                "AnisotropicDiffusion: empty brain mask for %s, returning input",
                volume.path,
            )
            return volume
        lo = float(np.percentile(in_brain, self.percentile_clip[0]))
        hi = float(np.percentile(in_brain, self.percentile_clip[1]))
        if hi <= lo:
            logger.warning(
                "AnisotropicDiffusion: degenerate intensity range for %s, returning zeros",
                volume.path,
            )
            zeroed = np.zeros_like(arr, dtype=np.float32)
            return NiftiVolume(
                array=zeroed,
                affine=volume.affine,
                header=volume.header,
                path=volume.path,
                spacing_mm=volume.spacing_mm,
            )
        normalised = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
        normalised *= brain.astype(np.float32)

        # Hand the volume to ITK-TubeTK. itk.GetImageFromArray takes a (z, y, x)
        # ordered view; we transpose to match ITK's IndexValueType convention.
        itk_image = itk.GetImageFromArray(np.ascontiguousarray(normalised))
        try:
            enhanced_itk = itk.EnhanceTubesUsingDiffusion(
                itk_image,
                number_of_iterations=self.n_iterations,
                number_of_scales=self.n_scales,
                min_sigma=self.sigma_min,
                max_sigma=self.sigma_max,
                time_step=self.time_step,
            )
        except RuntimeError as exc:
            raise PreprocessingError(
                f"ITK-TubeTK EnhanceTubesUsingDiffusion failed on {volume.path}: {exc}"
            ) from exc
        enhanced = np.asarray(itk.GetArrayFromImage(enhanced_itk), dtype=np.float32)
        enhanced *= brain.astype(np.float32)

        return NiftiVolume(
            array=enhanced,
            affine=volume.affine,
            header=volume.header,
            path=volume.path,
            spacing_mm=volume.spacing_mm,
        )

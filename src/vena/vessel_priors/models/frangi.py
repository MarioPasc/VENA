"""Frangi vesselness baseline.

Implements proposal §3.2's *default path*: multi-scale Frangi filter
(`skimage.filters.frangi`) on the bias-corrected SWI volume with
``black_ridges=True``, σ ∈ [0.5, 2.5] mm. UCSF-PDGM is 1 mm isotropic so the
σ values can be used directly as voxel sigmas.

References
----------
Frangi, Niessen, Vincken, Viergever. *Multiscale vessel enhancement filtering*.
MICCAI 1998. https://doi.org/10.1007/BFb0056195

Morrison et al. *Reproducible cerebrovascular segmentation from SWI*. Scientific
Reports 11:4404, 2021. https://doi.org/10.1038/s41598-021-83607-0 (empirical
validation of Frangi on SWI).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from skimage.filters import frangi

from vena.vessel_priors.abc_model import (
    AbstractVesselModel,
    VesselInput,
    VesselOutput,
)

logger = logging.getLogger(__name__)


class FrangiVesselModel(AbstractVesselModel):
    """Multi-scale Frangi vesselness on a brain-masked SWI volume.

    Parameters
    ----------
    sigma_min_mm, sigma_max_mm
        Inclusive σ range in millimetres. Proposal default: ``[0.5, 2.5]``.
    sigma_steps
        Number of σ samples uniformly between min and max (default 5).
    black_ridges
        ``True`` selects dark tubular structures (SWI vessels). Proposal default.
    threshold
        Soft-response threshold used to derive the binary mask. Proposal default
        ``0.15`` (from ``preflight-pattern.md`` decision contract).
    normalize
        Intensity normalisation prior to filtering. ``"percentile"`` clips at
        ``percentile_clip`` then min-max-scales to ``[0, 1]`` within the brain
        mask. ``"minmax"`` skips the clip.
    percentile_clip
        Lower / upper percentile when ``normalize == "percentile"``.
    voxel_spacing_mm
        Optional ``(sx, sy, sz)``. When provided, σ values are converted from
        millimetres to voxels by dividing by the mean isotropic spacing. UCSF-
        PDGM is 1 mm isotropic so the default ``None`` is safe.
    """

    name: ClassVar[str] = "frangi"

    def __init__(
        self,
        sigma_min_mm: float = 0.5,
        sigma_max_mm: float = 2.5,
        sigma_steps: int = 5,
        black_ridges: bool = True,
        threshold: float = 0.15,
        normalize: Literal["minmax", "percentile"] = "percentile",
        percentile_clip: tuple[float, float] = (0.5, 99.5),
        voxel_spacing_mm: tuple[float, float, float] | None = None,
    ) -> None:
        if sigma_min_mm <= 0 or sigma_max_mm <= 0:
            raise ValueError("σ values must be positive")
        if sigma_max_mm < sigma_min_mm:
            raise ValueError("sigma_max_mm must be >= sigma_min_mm")
        if sigma_steps < 1:
            raise ValueError("sigma_steps must be >= 1")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must lie in [0, 1]")
        self.sigma_min_mm = float(sigma_min_mm)
        self.sigma_max_mm = float(sigma_max_mm)
        self.sigma_steps = int(sigma_steps)
        self.black_ridges = bool(black_ridges)
        self.threshold = float(threshold)
        self.normalize = normalize
        self.percentile_clip = (float(percentile_clip[0]), float(percentile_clip[1]))
        self.voxel_spacing_mm = (
            tuple(float(s) for s in voxel_spacing_mm) if voxel_spacing_mm else None
        )

    # ------------------------------------------------------------------ helpers

    def _sigmas(self, spacing_mm: tuple[float, float, float]) -> NDArray[np.float64]:
        # Use the cohort/voxel mean spacing to convert mm → voxels. UCSF-PDGM is
        # isotropic; for anisotropic data this is an approximation and a future
        # extension could pass an anisotropic σ per-axis if `skimage` supports it.
        s = self.voxel_spacing_mm or spacing_mm
        mean_mm = float(np.mean(s))
        sig_mm = np.linspace(self.sigma_min_mm, self.sigma_max_mm, self.sigma_steps)
        return sig_mm / mean_mm

    def _normalize(
        self,
        img: NDArray[np.float32],
        brain: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        in_brain = img[brain]
        if in_brain.size == 0:
            return img
        if self.normalize == "percentile":
            lo = float(np.percentile(in_brain, self.percentile_clip[0]))
            hi = float(np.percentile(in_brain, self.percentile_clip[1]))
        else:
            lo = float(in_brain.min())
            hi = float(in_brain.max())
        if hi <= lo:
            return np.zeros_like(img)
        out = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
        out *= brain.astype(np.float32)
        return out

    # ------------------------------------------------------------------ predict

    def predict(self, x: VesselInput) -> VesselOutput:
        swi = x.swi
        brain = (x.brain_mask > 0).astype(bool)
        if brain.shape != swi.array.shape:
            raise ValueError(
                f"brain mask shape {brain.shape} != SWI shape {swi.array.shape}"
            )

        img = swi.array.astype(np.float32, copy=False)
        normalised = self._normalize(img, brain)

        sigmas = self._sigmas(swi.spacing_mm)
        logger.info(
            "Frangi: pid=%s sigmas(vox)=%s black_ridges=%s",
            x.patient_id,
            np.round(sigmas, 3).tolist(),
            self.black_ridges,
        )

        raw = frangi(
            normalised,
            sigmas=sigmas,
            black_ridges=self.black_ridges,
        ).astype(np.float32)

        # Constrain response to the brain interior and rescale to [0, 1] so
        # downstream thresholds and visualisations are comparable across patients.
        raw *= brain.astype(np.float32)
        peak = float(raw.max())
        soft = raw / (peak + 1e-8) if peak > 0 else raw

        binary = ((soft >= self.threshold) & brain).astype(np.uint8)

        params: dict[str, Any] = {
            "sigma_min_mm": self.sigma_min_mm,
            "sigma_max_mm": self.sigma_max_mm,
            "sigma_steps": self.sigma_steps,
            "black_ridges": self.black_ridges,
            "threshold": self.threshold,
            "normalize": self.normalize,
            "percentile_clip": list(self.percentile_clip),
            "voxel_spacing_mm": list(swi.spacing_mm),
            "sigmas_voxel": [float(v) for v in sigmas],
            "raw_response_peak": peak,
        }

        self._validate_output(soft, binary, swi)

        return VesselOutput(
            soft=soft.astype(np.float32),
            binary=binary,
            affine=swi.affine.copy(),
            threshold=self.threshold,
            params=params,
        )

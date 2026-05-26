"""SWAN magnitude → susceptibility surrogate conditioning channels.

Sub-option A of `soft_priors_sources.md` §4.3 (no phase data available). Two
channels are derived from the magnitude alone:

* ``sus``  = ``G_sigma(1 - percentile_norm(SWAN))`` — smoothed darkness field.
* ``itss`` = ``M_tum * G_sigma(1[ SWAN < q10(SWAN | M_tum) ])`` — smoothed
  in-tumour ITSS density (Pinker 2009 quantitative definition).

The kernel σ is interpreted in millimetres and converted to a voxel σ via the
input's spacing (`scipy.ndimage.gaussian_filter` operates in voxels).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from vena.prior_maps.susceptibility_priors.abc_model import (
    AbstractSusceptibilityModel,
    PriorOutput,
    SusceptibilityInput,
)

logger = logging.getLogger(__name__)


def _percentile_norm(
    arr: NDArray[np.float32],
    mask: NDArray[np.bool_],
    lo_pct: float,
    hi_pct: float,
) -> NDArray[np.float32]:
    in_mask = arr[mask]
    if in_mask.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(in_mask, lo_pct))
    hi = float(np.percentile(in_mask, hi_pct))
    if hi <= lo:
        # Percentile range collapsed (e.g. a sparse minority of outlier voxels
        # below the bulk distribution). Fall back to min/max so the contrast
        # information is preserved instead of dropping the whole field to 0.
        lo = float(in_mask.min())
        hi = float(in_mask.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


class MagnitudeSwanSusceptibilityModel(AbstractSusceptibilityModel):
    """Magnitude-only susceptibility channels from SWAN.

    Parameters
    ----------
    sigma_mm
        Gaussian smoothing radius for both channels, in millimetres. Default
        ``1.0`` per `soft_priors_sources.md` §4.3 sub-option A.
    percentile_clip
        Percentile clip for ``percentile_norm``. Default ``(0.5, 99.5)``.
    itss_quantile
        Quantile used to define the dark-foci indicator inside the tumour.
        Default ``0.10`` per Pinker 2009.
    contour_threshold
        Soft-response threshold used to derive the (collage-only) binary
        contour from ``sus``. Default ``0.5``.
    """

    name: ClassVar[str] = "magnitude_swan"

    def __init__(
        self,
        sigma_mm: float = 1.0,
        percentile_clip: tuple[float, float] = (0.5, 99.5),
        itss_quantile: float = 0.10,
        contour_threshold: float = 0.5,
    ) -> None:
        if sigma_mm <= 0:
            raise ValueError("sigma_mm must be positive")
        if not (0.0 < itss_quantile < 1.0):
            raise ValueError("itss_quantile must lie in (0, 1)")
        if not (0.0 <= contour_threshold <= 1.0):
            raise ValueError("contour_threshold must lie in [0, 1]")
        self.sigma_mm = float(sigma_mm)
        self.percentile_clip = (float(percentile_clip[0]), float(percentile_clip[1]))
        self.itss_quantile = float(itss_quantile)
        self.contour_threshold = float(contour_threshold)

    def predict(self, x: SusceptibilityInput) -> PriorOutput:
        swan = x.swi.array.astype(np.float32, copy=False)
        brain = (x.brain_mask > 0).astype(bool)
        tumour = (x.tumour_mask > 0).astype(bool)
        for arr, label in ((brain, "brain_mask"), (tumour, "tumour_mask")):
            if arr.shape != swan.shape:
                raise ValueError(f"{label} shape {arr.shape} != SWAN shape {swan.shape}")

        # Voxel-space σ per axis, derived from the volume's physical spacing.
        sx, sy, sz = x.swi.spacing_mm
        sigma_vox = (self.sigma_mm / sx, self.sigma_mm / sy, self.sigma_mm / sz)

        in_brain = swan[brain]
        if in_brain.size == 0:
            dynamic_range = 0.0
        else:
            dynamic_range = float(in_brain.max() - in_brain.min())
        if dynamic_range < 1e-6:
            # Degenerate (uniform) input — no susceptibility contrast carries
            # information; the (1 - percentile_norm) construction would
            # otherwise spuriously flag every voxel as maximally dark.
            logger.warning(
                "magnitude_swan: degenerate SWAN dynamic range for %s — sus = 0.",
                x.patient_id,
            )
            sus = np.zeros_like(swan, dtype=np.float32)
        else:
            norm = _percentile_norm(swan, brain, self.percentile_clip[0], self.percentile_clip[1])
            darkness = (1.0 - norm).astype(np.float32)
            darkness *= brain.astype(np.float32)
            sus = gaussian_filter(darkness, sigma=sigma_vox).astype(np.float32)
            sus *= brain.astype(np.float32)
            # Clip tiny negatives from filtering edge effects.
            sus = np.clip(sus, 0.0, None).astype(np.float32)

        # In-tumour ITSS: indicator of voxels below the in-tumour 10th
        # percentile, smoothed and gated to the tumour mask.
        if tumour.any():
            in_tumour = swan[tumour]
            q10 = float(np.quantile(in_tumour, self.itss_quantile))
            indicator = ((swan < q10) & tumour).astype(np.float32)
        else:
            logger.warning(
                "magnitude_swan: empty tumour mask for %s — ITSS = 0 everywhere.",
                x.patient_id,
            )
            q10 = float("nan")
            indicator = np.zeros_like(swan, dtype=np.float32)
        itss = gaussian_filter(indicator, sigma=sigma_vox).astype(np.float32)
        itss *= tumour.astype(np.float32)
        itss = np.clip(itss, 0.0, None).astype(np.float32)

        binary = ((sus >= self.contour_threshold) & brain).astype(np.uint8)

        params: dict[str, Any] = {
            "sigma_mm": self.sigma_mm,
            "sigma_voxels": list(sigma_vox),
            "percentile_clip": list(self.percentile_clip),
            "itss_quantile": self.itss_quantile,
            "contour_threshold": self.contour_threshold,
            "itss_q10_value": q10,
            "tumour_voxel_count": int(tumour.sum()),
            "voxel_spacing_mm": list(x.swi.spacing_mm),
        }

        out = PriorOutput(
            channels={"sus": sus, "itss": itss},
            binary=binary,
            affine=x.swi.affine.copy(),
            params=params,
        )
        self._validate_output(out, x.swi)
        return out

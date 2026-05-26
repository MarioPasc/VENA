"""ADC → cellularity conditioning channels via NAWM normalisation.

Implements `soft_priors_sources.md` §4.2: starting from a quantitative ADC map
(post Stejskal–Tanner fit; UCSF-PDGM ships ADC at the T1 isotropic grid),
build two conditioning channels.

Pipeline:

1. NAWM median :math:`\\overline{\\text{ADC}}_{\\text{NAWM}}` over parenchyma
   minus tumour.
2. ``adc_rel = clip(ADC / max(median, eps), 0, max_relative)``.
3. ``cell = M_tum * sigmoid((median - ADC) / sigma_adc)`` with
   :math:`\\sigma_{\\text{ADC}}` defined as a fraction of the NAWM median so
   the channel is dimensionless and scanner-agnostic.

Per §4.2, the *raw* whole-brain ADC failed in Preetha 2021 because it has no
tumour-locality signal; the ``cell`` channel restores that signal by gating
the restricted-diffusion indicator with :math:`M_{\\text{tum}}`.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from vena.prior_maps.cellularity_priors.abc_model import (
    AbstractCellularityModel,
    CellularityInput,
    PriorOutput,
)

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class NAWMNormalizedCellularityModel(AbstractCellularityModel):
    """NAWM-normalised ADC channels and tumour-gated cellularity indicator.

    Parameters
    ----------
    max_relative
        Upper clip for ``adc_rel``. Default ``4.0`` matches the
        :math:`[0.3, 4]` range stated in `soft_priors_sources.md` §4.2.
    sigma_adc_fraction
        Width of the sigmoid relative to the NAWM median. Default ``0.2`` so
        a 20% reduction below NAWM yields ``cell ≈ 0.62`` inside the tumour
        (§4.2 conjecture; tunable).
    eps
        Stabiliser added to the NAWM median.
    aggregator
        ``"median"`` (default) or ``"mean"``.
    """

    name: ClassVar[str] = "nawm_normalized"

    def __init__(
        self,
        max_relative: float = 4.0,
        sigma_adc_fraction: float = 0.2,
        eps: float = 1e-9,
        aggregator: str = "median",
    ) -> None:
        if max_relative <= 0:
            raise ValueError("max_relative must be positive")
        if sigma_adc_fraction <= 0:
            raise ValueError("sigma_adc_fraction must be positive")
        if aggregator not in ("median", "mean"):
            raise ValueError("aggregator must be 'median' or 'mean'")
        self.max_relative = float(max_relative)
        self.sigma_adc_fraction = float(sigma_adc_fraction)
        self.eps = float(eps)
        self.aggregator = aggregator

    def predict(self, x: CellularityInput) -> PriorOutput:
        adc_raw = x.adc.array.astype(np.float32, copy=False)
        brain = (x.brain_mask > 0).astype(bool)
        parenchyma = (x.parenchyma_mask > 0).astype(bool)
        tumour = (x.tumour_mask > 0).astype(bool)
        for arr, label in (
            (brain, "brain_mask"),
            (parenchyma, "parenchyma_mask"),
            (tumour, "tumour_mask"),
        ):
            if arr.shape != adc_raw.shape:
                raise ValueError(f"{label} shape {arr.shape} != ADC shape {adc_raw.shape}")

        nawm = parenchyma & (~tumour)
        nawm_voxels = adc_raw[nawm]
        if nawm_voxels.size == 0:
            logger.warning(
                "Cellularity: empty NAWM for %s — falling back to brain median.",
                x.patient_id,
            )
            nawm_voxels = adc_raw[brain]
        if nawm_voxels.size == 0:
            raise ValueError(f"Empty brain mask for {x.patient_id}")

        if self.aggregator == "median":
            adc_nawm = float(np.median(nawm_voxels))
        else:
            adc_nawm = float(np.mean(nawm_voxels))
        if adc_nawm <= self.eps:
            fallback = float(np.percentile(adc_raw[brain], 50.0))
            logger.warning(
                "Cellularity: NAWM median %.4g <= eps for %s; fallback p50 = %.4g",
                adc_nawm,
                x.patient_id,
                fallback,
            )
            adc_nawm = max(fallback, self.eps)

        adc_rel = adc_raw / adc_nawm
        adc_rel = np.clip(adc_rel, 0.0, self.max_relative).astype(np.float32)
        adc_rel *= brain.astype(np.float32)

        sigma_adc_abs = self.sigma_adc_fraction * adc_nawm
        # cell ∈ [0, 1]; the sigmoid argument is positive where ADC is below
        # the NAWM median (restricted diffusion → cellular tumour).
        cell_full = _sigmoid((adc_nawm - adc_raw) / max(sigma_adc_abs, self.eps))
        cell = (cell_full * tumour.astype(np.float32)).astype(np.float32)

        params: dict[str, Any] = {
            "max_relative": self.max_relative,
            "sigma_adc_fraction": self.sigma_adc_fraction,
            "sigma_adc_absolute": sigma_adc_abs,
            "aggregator": self.aggregator,
            "adc_nawm_reference": adc_nawm,
            "nawm_voxel_count": int(nawm.sum()),
            "tumour_voxel_count": int(tumour.sum()),
            "voxel_spacing_mm": list(x.adc.spacing_mm),
        }

        out = PriorOutput(
            channels={"adc_rel": adc_rel, "cell": cell},
            binary=tumour.astype(np.uint8),
            affine=x.adc.affine.copy(),
            params=params,
        )
        self._validate_output(out, x.adc)
        return out

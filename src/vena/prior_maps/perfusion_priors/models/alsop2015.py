"""ASL → CBF conditioning channels via NAWM normalisation.

Implements `soft_priors_sources.md` §4.1: starting from a quantitative CBF map
(post-ASLPrep, single-PLD pCASL per the Alsop 2015 consensus), build two
conditioning channels by referencing the NAWM median.

The pipeline is:

1. Compute the NAWM median :math:`\\overline{\\mathrm{CBF}}_{\\text{NAWM}}` over
   parenchyma minus tumour.
2. ``cbf_rel = clip(CBF / max(median, eps), 0, max_relative)`` — removes
   scanner / age effects; NAWM lands at 1.
3. ``cbf = tanh(cbf_rel / squash_const)`` — bounded to :math:`[-1, 1]` for
   numerical stability with the MAISI-trained trunk.

The "Alsop 2015" name is used here for traceability of the upstream pipeline
that produced the CBF map; the *transformation* this module applies is a
NAWM-relative normalisation, not the Alsop quantification itself.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from vena.prior_maps.perfusion_priors.abc_model import (
    AbstractPerfusionModel,
    PerfusionInput,
    PriorOutput,
)

logger = logging.getLogger(__name__)


class Alsop2015PerfusionModel(AbstractPerfusionModel):
    """NAWM-normalised CBF channels from a quantitative ASL CBF map.

    Parameters
    ----------
    max_relative
        Upper clip for ``cbf_rel``. Default ``8.0`` matches the
        :math:`\\sim [0, 8]` range stated in `soft_priors_sources.md` §4.1.
    squash_const
        Divisor inside ``tanh`` applied to ``cbf_rel``. Default ``3.0``
        (NAWM @ 1 → ``cbf = tanh(1/3) ≈ 0.32``; HG-tumour core @ 5 → ``≈ 0.93``).
    eps
        Stabiliser added to the NAWM median before division.
    aggregator
        ``"median"`` (default; robust to outliers) or ``"mean"``.
    """

    name: ClassVar[str] = "alsop2015"

    def __init__(
        self,
        max_relative: float = 8.0,
        squash_const: float = 3.0,
        eps: float = 1e-6,
        aggregator: str = "median",
    ) -> None:
        if max_relative <= 0:
            raise ValueError("max_relative must be positive")
        if squash_const <= 0:
            raise ValueError("squash_const must be positive")
        if aggregator not in ("median", "mean"):
            raise ValueError(f"aggregator must be 'median' or 'mean', got {aggregator!r}")
        self.max_relative = float(max_relative)
        self.squash_const = float(squash_const)
        self.eps = float(eps)
        self.aggregator = aggregator

    def predict(self, x: PerfusionInput) -> PriorOutput:
        cbf_raw = x.asl.array.astype(np.float32, copy=False)
        brain = (x.brain_mask > 0).astype(bool)
        parenchyma = (x.parenchyma_mask > 0).astype(bool)
        tumour = (x.tumour_mask > 0).astype(bool)
        for arr, label in (
            (brain, "brain_mask"),
            (parenchyma, "parenchyma_mask"),
            (tumour, "tumour_mask"),
        ):
            if arr.shape != cbf_raw.shape:
                raise ValueError(f"{label} shape {arr.shape} != ASL shape {cbf_raw.shape}")

        nawm = parenchyma & (~tumour)
        nawm_voxels = cbf_raw[nawm]
        if nawm_voxels.size == 0:
            logger.warning(
                "Alsop2015: empty NAWM for %s — falling back to whole-brain median.",
                x.patient_id,
            )
            nawm_voxels = cbf_raw[brain]
        if nawm_voxels.size == 0:
            raise ValueError(f"Empty brain mask for {x.patient_id}")
        if self.aggregator == "median":
            cbf_nawm = float(np.median(nawm_voxels))
        else:
            cbf_nawm = float(np.mean(nawm_voxels))
        if cbf_nawm <= self.eps:
            # Negative or near-zero CBF reference happens with noisy ASL; fall
            # back to absolute scaling using the brain-wide 95th percentile so
            # the downstream tanh stays well-conditioned.
            fallback = float(np.percentile(cbf_raw[brain], 95.0))
            logger.warning(
                "Alsop2015: NAWM median %.4g <= eps for %s; falling back to brain-wide p95 = %.4g",
                cbf_nawm,
                x.patient_id,
                fallback,
            )
            cbf_nawm = max(fallback, self.eps)

        cbf_rel = cbf_raw / cbf_nawm
        cbf_rel = np.clip(cbf_rel, 0.0, self.max_relative).astype(np.float32)
        cbf_rel *= brain.astype(np.float32)

        cbf = np.tanh(cbf_rel / self.squash_const).astype(np.float32)
        # Suppress non-brain voxels so the ControlNet branch sees a clean
        # background of exactly 0.
        cbf *= brain.astype(np.float32)

        params: dict[str, Any] = {
            "max_relative": self.max_relative,
            "squash_const": self.squash_const,
            "aggregator": self.aggregator,
            "cbf_nawm_reference": cbf_nawm,
            "nawm_voxel_count": int(nawm.sum()),
            "voxel_spacing_mm": list(x.asl.spacing_mm),
        }

        out = PriorOutput(
            channels={"cbf_rel": cbf_rel, "cbf": cbf},
            binary=None,
            affine=x.asl.affine.copy(),
            params=params,
        )
        self._validate_output(out, x.asl)
        return out

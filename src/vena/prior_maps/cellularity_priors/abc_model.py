"""Abstract base class for cellularity-prior extraction models.

A cellularity-prior model consumes a quantitative ADC map (already produced by
a Stejskal–Tanner DWI fit; UCSF-PDGM v4 ships ADC at the T1 isotropic grid)
plus brain / parenchyma / tumour masks and emits two conditioning channels
that constrain term (II) of the Tofts–Kermode decomposition
(`soft_priors_sources.md` §4.2):

* ``adc_rel`` — ADC normalised by the NAWM median; range :math:`\\sim [0.3, 4]`.
* ``cell`` — tumour-restricted cellularity indicator
  :math:`M_{\\text{tum}} \\cdot \\sigma\\big((\\overline{\\text{ADC}}_{\\text{NAWM}} - \\text{ADC}) / \\sigma_{\\text{ADC}}\\big)`
  in :math:`[0, 1]`, peaked where ADC is restricted *within* the tumour mask.

Concrete implementations live under
:mod:`vena.prior_maps.cellularity_priors.models`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from vena.data.niigz import NiftiVolume

REQUIRED_CHANNELS: tuple[str, ...] = ("adc_rel", "cell")


class CellularityPriorError(Exception):
    """Raised at the model boundary for shape/dtype/affine contract violations."""


@dataclass(frozen=True)
class CellularityInput:
    """Input bundle for :meth:`AbstractCellularityModel.predict`.

    Parameters
    ----------
    adc
        Quantitative ADC map (mm²/s; ~``[0, 5e-3]`` on UCSF-PDGM).
    brain_mask
        Binary brain mask.
    parenchyma_mask
        Binary parenchyma segmentation (white + grey matter); used together
        with ``tumour_mask`` to define the NAWM proxy.
    tumour_mask
        Binary tumour mask (UCSF-PDGM ``tumor_segmentation > 0``); gates
        ``cell`` to the support of term (II).
    patient_id
        Cohort identifier, propagated into logs and the manifest.
    """

    adc: NiftiVolume
    brain_mask: NDArray[Any]
    parenchyma_mask: NDArray[Any]
    tumour_mask: NDArray[Any]
    patient_id: str


@dataclass(frozen=True)
class PriorOutput:
    """Output bundle returned from :meth:`AbstractCellularityModel.predict`."""

    channels: dict[str, NDArray[np.float32]]
    binary: NDArray[np.uint8] | None
    affine: NDArray[np.floating[Any]]
    params: dict[str, Any] = field(default_factory=dict)


class AbstractCellularityModel(ABC):
    """Contract for any cellularity-prior model used in this project."""

    name: ClassVar[str]

    @abstractmethod
    def __init__(self, **params: Any) -> None: ...

    @abstractmethod
    def predict(self, x: CellularityInput) -> PriorOutput: ...

    @staticmethod
    def _validate_output(out: PriorOutput, reference: NiftiVolume) -> None:
        for key in REQUIRED_CHANNELS:
            if key not in out.channels:
                raise CellularityPriorError(f"Missing required channel: {key!r}")
            arr = out.channels[key]
            if arr.shape != reference.array.shape:
                raise CellularityPriorError(
                    f"channel {key!r} shape {arr.shape} != input {reference.array.shape}"
                )
            if arr.dtype != np.float32:
                raise CellularityPriorError(
                    f"channel {key!r} dtype must be float32, got {arr.dtype}"
                )
            if not np.isfinite(arr).all():
                raise CellularityPriorError(f"channel {key!r} contains non-finite values")
        cell = out.channels["cell"]
        if float(cell.min()) < -1e-6 or float(cell.max()) > 1.0 + 1e-6:
            raise CellularityPriorError(
                f"channel 'cell' must lie in [0, 1], got [{cell.min()}, {cell.max()}]"
            )
        if float(out.channels["adc_rel"].min()) < 0.0 - 1e-6:
            raise CellularityPriorError("channel 'adc_rel' must be non-negative")
        if out.binary is not None and out.binary.dtype != np.uint8:
            raise CellularityPriorError(f"binary dtype must be uint8, got {out.binary.dtype}")

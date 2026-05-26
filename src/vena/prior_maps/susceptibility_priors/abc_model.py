"""Abstract base class for susceptibility-prior extraction models.

A susceptibility-prior model consumes a SWI / SWAN magnitude volume and the
tumour mask, and emits two conditioning channels per
`soft_priors_sources.md` §4.3 sub-option A:

* ``sus`` — smoothed "darkness" field
  :math:`G_\\sigma(1 - \\text{percentile-norm}(\\text{SWAN}))`, a continuous
  susceptibility surrogate that lights up veins, hemorrhage, and iron.
* ``itss`` — tumour-restricted ITSS density (smoothed indicator of voxels
  below the in-tumour 10th percentile).

This is the *magnitude-only* sub-option A. Sub-option B (QSM, phase data
required) and sub-option C (multi-echo χ-separation) are deferred until HRUM
exports phase alongside magnitude.

Concrete implementations live under
:mod:`vena.prior_maps.susceptibility_priors.models`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from vena.data.niigz import NiftiVolume

REQUIRED_CHANNELS: tuple[str, ...] = ("sus", "itss")


class SusceptibilityPriorError(Exception):
    """Raised at the model boundary for shape/dtype contract violations."""


@dataclass(frozen=True)
class SusceptibilityInput:
    """Input bundle for :meth:`AbstractSusceptibilityModel.predict`.

    Parameters
    ----------
    swi
        SWI / SWAN magnitude volume (bias-corrected if available).
    brain_mask
        Binary brain mask.
    tumour_mask
        Binary tumour mask; gates the ``itss`` channel.
    patient_id
        Cohort identifier.
    """

    swi: NiftiVolume
    brain_mask: NDArray[Any]
    tumour_mask: NDArray[Any]
    patient_id: str


@dataclass(frozen=True)
class PriorOutput:
    channels: dict[str, NDArray[np.float32]]
    binary: NDArray[np.uint8] | None
    affine: NDArray[np.floating[Any]]
    params: dict[str, Any] = field(default_factory=dict)


class AbstractSusceptibilityModel(ABC):
    """Contract for any susceptibility-prior model used in this project."""

    name: ClassVar[str]

    @abstractmethod
    def __init__(self, **params: Any) -> None: ...

    @abstractmethod
    def predict(self, x: SusceptibilityInput) -> PriorOutput: ...

    @staticmethod
    def _validate_output(out: PriorOutput, reference: NiftiVolume) -> None:
        for key in REQUIRED_CHANNELS:
            if key not in out.channels:
                raise SusceptibilityPriorError(f"Missing required channel: {key!r}")
            arr = out.channels[key]
            if arr.shape != reference.array.shape:
                raise SusceptibilityPriorError(
                    f"channel {key!r} shape {arr.shape} != input {reference.array.shape}"
                )
            if arr.dtype != np.float32:
                raise SusceptibilityPriorError(
                    f"channel {key!r} dtype must be float32, got {arr.dtype}"
                )
            if not np.isfinite(arr).all():
                raise SusceptibilityPriorError(f"channel {key!r} contains non-finite values")
            if float(arr.min()) < -1e-6:
                raise SusceptibilityPriorError(
                    f"channel {key!r} must be non-negative, got min={arr.min()}"
                )
        if out.binary is not None and out.binary.dtype != np.uint8:
            raise SusceptibilityPriorError(f"binary dtype must be uint8, got {out.binary.dtype}")

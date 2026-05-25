"""Abstract base class for vessel-prior extraction models.

A vessel-prior model consumes a single brain volume (typically a bias-corrected
SWI / SWAN scan together with its binary brain mask) and emits a vesselness
response in the *same physical space* as the input. Concrete implementations
live under :mod:`vena.vessel_priors.models` and are registered in
:data:`vena.vessel_priors.models.MODEL_REGISTRY`.

Output contract (enforced by the engine):

* ``soft``   : float32 array, shape == input shape, values in ``[0, 1]``.
* ``binary`` : uint8 array, shape == input shape, values in ``{0, 1}``.
* ``affine`` : 4x4 matching the input volume's affine bit-for-bit.

This keeps the routine, the saver, and the collage renderer agnostic to the
specific vesselness algorithm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from vena.data.niigz import NiftiVolume


class VesselPriorError(Exception):
    """Raised at the model boundary for shape/dtype/affine contract violations."""


@dataclass(frozen=True)
class VesselInput:
    """Input bundle handed to :meth:`AbstractVesselModel.predict`.

    Parameters
    ----------
    swi
        Bias-corrected SWI / SWAN volume.
    brain_mask
        Binary brain mask, ``bool`` or ``{0, 1}``-valued numeric array. Same
        shape as ``swi.array``. Used to suppress vesselness response in
        non-brain regions.
    patient_id
        Cohort patient identifier, propagated into logs and the manifest.
    """

    swi: NiftiVolume
    brain_mask: NDArray[Any]
    patient_id: str


@dataclass(frozen=True)
class VesselOutput:
    """Output bundle returned from :meth:`AbstractVesselModel.predict`.

    Parameters
    ----------
    soft
        Float32 vesselness response in ``[0, 1]``, same shape as the input.
    binary
        Uint8 thresholded mask, same shape as the input.
    affine
        4x4 voxel-to-world transform inherited from the input volume.
    threshold
        Threshold value applied to ``soft`` to produce ``binary``.
    params
        Snapshot of the model parameters that produced this output. Round-trips
        into the per-run manifest.
    """

    soft: NDArray[np.float32]
    binary: NDArray[np.uint8]
    affine: NDArray[np.floating[Any]]
    threshold: float
    params: dict[str, Any] = field(default_factory=dict)


class AbstractVesselModel(ABC):
    """Contract for any vessel-prior model used in this project."""

    name: ClassVar[str]
    """Registry key (e.g. ``"frangi"``). Concrete classes must set this."""

    @abstractmethod
    def __init__(self, **params: Any) -> None: ...

    @abstractmethod
    def predict(self, x: VesselInput) -> VesselOutput:
        """Run the vesselness operator on ``x`` and return the response bundle."""

    # ----- helpers for subclasses ---------------------------------------------

    @staticmethod
    def _validate_output(
        soft: NDArray[Any],
        binary: NDArray[Any],
        reference: NiftiVolume,
    ) -> None:
        """Assert the output respects the contract documented at module level."""
        if soft.shape != reference.array.shape:
            raise VesselPriorError(
                f"soft shape {soft.shape} != input shape {reference.array.shape}"
            )
        if binary.shape != reference.array.shape:
            raise VesselPriorError(
                f"binary shape {binary.shape} != input shape {reference.array.shape}"
            )
        if soft.dtype != np.float32:
            raise VesselPriorError(f"soft dtype must be float32, got {soft.dtype}")
        if binary.dtype != np.uint8:
            raise VesselPriorError(f"binary dtype must be uint8, got {binary.dtype}")
        smin, smax = float(soft.min()), float(soft.max())
        if smin < 0.0 or smax > 1.0 + 1e-6:
            raise VesselPriorError(f"soft must lie in [0, 1], got [{smin}, {smax}]")

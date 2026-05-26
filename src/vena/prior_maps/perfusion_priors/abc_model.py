"""Abstract base class for perfusion-prior extraction models.

A perfusion-prior model consumes a quantitative CBF map (already produced by an
ASLPrep-equivalent pipeline) plus brain / parenchyma / tumour masks and emits a
pair of conditioning channels for the latent flow-matching trunk:

* ``cbf_rel`` — CBF normalised by the NAWM median (proxy for the
  normal-appearing white matter reference of `soft_priors_sources.md` §4.1).
  Range :math:`\\sim [0, 8]`.
* ``cbf`` — :math:`\\tanh(c^{\\text{rel}} / 3)` so the channel is bounded to
  :math:`[-1, 1]` for numerical stability with the MAISI-trained trunk
  (`soft_priors_sources.md` §4.1 squashing constant 3).

Concrete implementations live under
:mod:`vena.prior_maps.perfusion_priors.models` and are registered in
:data:`vena.prior_maps.perfusion_priors.models.MODEL_REGISTRY`.

Output contract (enforced by the engine):

* ``channels["cbf_rel"]``, ``channels["cbf"]``: float32, shape == input shape.
* ``binary`` : None (perfusion has no natural threshold).
* ``affine`` : 4x4 matching the input volume's affine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from vena.data.niigz import NiftiVolume

REQUIRED_CHANNELS: tuple[str, ...] = ("cbf_rel", "cbf")


class PerfusionPriorError(Exception):
    """Raised at the model boundary for shape/dtype/affine contract violations."""


@dataclass(frozen=True)
class PerfusionInput:
    """Input bundle for :meth:`AbstractPerfusionModel.predict`.

    Parameters
    ----------
    asl
        Quantitative CBF map (single 3D volume; UCSF-PDGM v4 ships post-ASLPrep
        CBF at the T1 isotropic grid).
    brain_mask
        Binary brain mask (same shape as ``asl.array``).
    parenchyma_mask
        Binary parenchyma segmentation (white + grey matter); used to build the
        NAWM proxy together with ``tumour_mask``.
    tumour_mask
        Binary tumour mask (UCSF-PDGM ``tumor_segmentation > 0``).
    patient_id
        Cohort identifier, propagated into logs and the manifest.
    """

    asl: NiftiVolume
    brain_mask: NDArray[Any]
    parenchyma_mask: NDArray[Any]
    tumour_mask: NDArray[Any]
    patient_id: str


@dataclass(frozen=True)
class PriorOutput:
    """Output bundle returned from :meth:`AbstractPerfusionModel.predict`.

    Parameters
    ----------
    channels
        Mapping ``channel_name -> float32 array``. Must contain the keys in
        :data:`REQUIRED_CHANNELS`.
    binary
        Optional uint8 mask for collage contours; ``None`` is valid.
    affine
        4x4 voxel-to-world transform inherited from the source volume.
    params
        Snapshot of the model parameters that produced this output.
    """

    channels: dict[str, NDArray[np.float32]]
    binary: NDArray[np.uint8] | None
    affine: NDArray[np.floating[Any]]
    params: dict[str, Any] = field(default_factory=dict)


class AbstractPerfusionModel(ABC):
    """Contract for any perfusion-prior model used in this project."""

    name: ClassVar[str]
    """Registry key (e.g. ``"alsop2015"``). Concrete classes must set this."""

    @abstractmethod
    def __init__(self, **params: Any) -> None: ...

    @abstractmethod
    def predict(self, x: PerfusionInput) -> PriorOutput:
        """Run the perfusion-prior derivation on ``x`` and return the bundle."""

    @staticmethod
    def _validate_output(
        out: PriorOutput,
        reference: NiftiVolume,
    ) -> None:
        """Assert the output respects the contract documented at module level."""
        for key in REQUIRED_CHANNELS:
            if key not in out.channels:
                raise PerfusionPriorError(f"Missing required channel: {key!r}")
            arr = out.channels[key]
            if arr.shape != reference.array.shape:
                raise PerfusionPriorError(
                    f"channel {key!r} shape {arr.shape} != input {reference.array.shape}"
                )
            if arr.dtype != np.float32:
                raise PerfusionPriorError(f"channel {key!r} dtype must be float32, got {arr.dtype}")
            if not np.isfinite(arr).all():
                raise PerfusionPriorError(f"channel {key!r} contains non-finite values")
        # Range checks per `soft_priors_sources.md` §4.1.
        cbf = out.channels["cbf"]
        if float(cbf.min()) < -1.0 - 1e-6 or float(cbf.max()) > 1.0 + 1e-6:
            raise PerfusionPriorError(
                f"channel 'cbf' must lie in [-1, 1], got [{cbf.min()}, {cbf.max()}]"
            )
        if float(out.channels["cbf_rel"].min()) < 0.0 - 1e-6:
            raise PerfusionPriorError("channel 'cbf_rel' must be non-negative")
        if out.binary is not None and out.binary.dtype != np.uint8:
            raise PerfusionPriorError(f"binary dtype must be uint8, got {out.binary.dtype}")

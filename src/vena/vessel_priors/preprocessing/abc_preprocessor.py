"""Abstract base class for SWI preprocessing applied before a vessel-prior model.

A preprocessor consumes a :class:`NiftiVolume` plus its brain mask and emits a
new :class:`NiftiVolume` in the *same physical space* (same affine, same shape,
same spacing). The brain mask is passed in so processors that benefit from a
restricted-histogram view (CLAHE, percentile clipping, ...) can ignore voxels
outside the brain.

Concrete implementations register themselves in
:mod:`vena.vessel_priors.preprocessing` so the engine can resolve them by their
string ``name`` declared in the YAML config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from numpy.typing import NDArray

from vena.data.niigz import NiftiVolume


class PreprocessingError(Exception):
    """Raised at the preprocessor boundary when the output contract is violated."""


class AbstractPreprocessor(ABC):
    """Contract for any SWI preprocessor used before a vessel-prior model."""

    name: ClassVar[str]
    """Registry key, set on concrete subclasses (e.g. ``"clahe"``)."""

    @abstractmethod
    def __init__(self, **params: Any) -> None: ...

    @abstractmethod
    def apply(self, volume: NiftiVolume, brain_mask: NDArray[Any]) -> NiftiVolume:
        """Return a new :class:`NiftiVolume` with the preprocessed array.

        Implementations MUST preserve ``volume.affine``, ``volume.header`` and
        ``volume.spacing_mm``. Only ``volume.array`` is replaced.
        """

    def describe(self) -> dict[str, Any]:
        """Snapshot of the parameters that drove this preprocessor.

        Used to round-trip the configuration into the per-run manifest. The
        default implementation reflects every attribute that doesn't begin
        with an underscore.
        """
        return {
            "name": self.name,
            "params": {k: v for k, v in self.__dict__.items() if not k.startswith("_")},
        }

"""Exception hierarchy for the vena.segmentation submodule.

Every public exception raised by library code under ``vena.segmentation``
descends from :class:`SegmentationError`.  Catching the base class is enough
for top-level handlers; catching a subclass narrows to a specific sub-system.
"""

from __future__ import annotations


class SegmentationError(Exception):
    """Base exception for all segmentation errors."""


class SegModelError(SegmentationError):
    """Raised when a model registry lookup fails or a model cannot be built."""


class SegDataError(SegmentationError):
    """Raised on data-loading, H5-schema, or fold-split violations."""


class SegLossError(SegmentationError):
    """Raised on loss configuration or computation errors."""


class SegTargetError(SegmentationError):
    """Raised when target generation (SDT, soft masks) fails."""


class SegDerivationError(SegmentationError):
    """Raised when latent-space mask derivation (pool, temperature) fails."""


class SegMetricError(SegmentationError):
    """Raised on metric computation or gate-check failures."""


__all__ = [
    "SegDataError",
    "SegDerivationError",
    "SegLossError",
    "SegMetricError",
    "SegModelError",
    "SegTargetError",
    "SegmentationError",
]

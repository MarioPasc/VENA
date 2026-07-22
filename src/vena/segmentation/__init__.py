"""VENA segmentation submodule.

Provides the frozen :class:`SegmentationConfig`, the decorator-based model
registry (:func:`register_segmentation_model`, :func:`get_segmentation_model`),
and the exception hierarchy (:class:`SegmentationError` and subclasses).

Wave-1 tasks fill the subpackages (``targets/``, ``data/``, ``engine/``,
``derivation/``, ``metrics/``); this ``__init__`` only re-exports the stable
contract needed by every downstream task.
"""

from __future__ import annotations

from vena.segmentation.config import (
    DataConfig,
    DerivationConfig,
    LossConfig,
    MetricsConfig,
    ModelConfig,
    SegmentationConfig,
    TargetConfig,
    TrainConfig,
)
from vena.segmentation.exceptions import (
    SegDataError,
    SegDerivationError,
    SegLossError,
    SegmentationError,
    SegMetricError,
    SegModelError,
    SegTargetError,
)
from vena.segmentation.models import (
    get_segmentation_model,
    register_segmentation_model,
    registered_model_names,
)

__all__ = [
    "DataConfig",
    "DerivationConfig",
    "LossConfig",
    "MetricsConfig",
    "ModelConfig",
    "SegDataError",
    "SegDerivationError",
    "SegLossError",
    "SegMetricError",
    "SegModelError",
    "SegTargetError",
    "SegmentationConfig",
    "SegmentationError",
    "TargetConfig",
    "TrainConfig",
    "get_segmentation_model",
    "register_segmentation_model",
    "registered_model_names",
]

"""Model registry for vena.segmentation.

Public API re-exported from :mod:`vena.segmentation.models.registry`.
"""

from __future__ import annotations

from vena.segmentation.models.registry import (
    get_segmentation_model,
    register_segmentation_model,
    registered_model_names,
)

__all__ = [
    "get_segmentation_model",
    "register_segmentation_model",
    "registered_model_names",
]

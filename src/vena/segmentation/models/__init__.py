"""Model registry for vena.segmentation.

Public API re-exported from :mod:`vena.segmentation.models.registry`.
"""

from __future__ import annotations

# Side-effect imports: trigger @register_segmentation_model decorators at
# package import time so all three arms are in the registry before any caller
# invokes get_segmentation_model().
from vena.segmentation.models import bsf_swinunetr, segresnet  # noqa: F401
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

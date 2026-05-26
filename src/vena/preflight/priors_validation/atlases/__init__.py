"""Atlas provisioning + registration for the priors-validation preflight."""

from __future__ import annotations

from .fetch import ensure_atlases
from .registry import ATLAS_REGISTRY, AtlasROI, atlas_label_value

__all__ = ["ATLAS_REGISTRY", "AtlasROI", "atlas_label_value", "ensure_atlases"]

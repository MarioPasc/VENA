"""Cohort-neutral latent-domain H5 manifest and converter (schema v2.0.0)."""

from .convert import LatentH5Config, LatentH5Converter
from .manifest import (
    LATENT_CHANNELS,
    LATENT_CROP_BOX,
    LATENT_SCHEMA_VERSION,
    LATENT_SEQUENCE_MAP,
    LATENT_SPATIAL,
    build_latent_manifest,
)

__all__ = [
    "LATENT_CHANNELS",
    "LATENT_CROP_BOX",
    "LATENT_SCHEMA_VERSION",
    "LATENT_SEQUENCE_MAP",
    "LATENT_SPATIAL",
    "LatentH5Config",
    "LatentH5Converter",
    "build_latent_manifest",
]

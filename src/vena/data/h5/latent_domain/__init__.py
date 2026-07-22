"""Cohort-neutral latent-domain H5 manifest and converter (schema v2.0.0).

Heavy converter classes (``LatentH5Converter``, ``LatentH5Config``) are NOT
re-exported here because loading them eagerly pulls in MAISI model code and
breaks import isolation for lightweight consumers (e.g.
``routines.segmentation.mask_derive``).  Import the converter directly when
needed::

    from vena.data.h5.latent_domain.convert import LatentH5Config, LatentH5Converter
"""

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
    "build_latent_manifest",
]

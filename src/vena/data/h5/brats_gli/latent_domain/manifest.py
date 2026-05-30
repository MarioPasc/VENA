"""BraTS-GLI thin manifest wrapper.

All neutral manifest logic lives in :mod:`vena.data.h5.latent_domain.manifest`.
BraTS-GLI carries no per-case clinical metadata (``metadata_fields=[]``).
"""

from __future__ import annotations

from vena.data.h5.latent_domain.manifest import (  # noqa: F401
    LATENT_SCHEMA_VERSION,
    LATENT_SPATIAL,
    build_latent_manifest,
)

BRATS_GLI_LATENT_DEFAULT_MODALITIES: list[str] = ["t1pre", "t1c", "t2", "flair"]

BRATS_GLI_LATENT_MANIFEST = build_latent_manifest(
    modalities=BRATS_GLI_LATENT_DEFAULT_MODALITIES,
    mask_output_channels=3,
    cohort="BraTS-GLI",
    metadata_fields=[],
)

__all__ = [
    "BRATS_GLI_LATENT_DEFAULT_MODALITIES",
    "BRATS_GLI_LATENT_MANIFEST",
    "LATENT_SCHEMA_VERSION",
    "LATENT_SPATIAL",
    "build_latent_manifest",
]

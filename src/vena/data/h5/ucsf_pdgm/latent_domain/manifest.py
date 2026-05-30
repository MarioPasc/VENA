"""UCSF-PDGM thin manifest wrapper (back-compat aliases + UCSF-specific constants).

All neutral manifest logic lives in :mod:`vena.data.h5.latent_domain.manifest`.
This module re-exports that logic under the legacy ``UCSF_PDGM_LATENT_*`` names
and adds the two UCSF-specific shape constants and the 15-field clinical metadata
list.  Downstream code that imports from this module continues to work unchanged.
"""

from __future__ import annotations

from vena.data.h5.latent_domain.manifest import (
    LATENT_CHANNELS,
    LATENT_SCHEMA_VERSION,
    LATENT_SEQUENCE_MAP,
    LATENT_SPATIAL,
    build_latent_manifest,
)

# Back-compat aliases — identical values, UCSF-prefixed names.
UCSF_PDGM_LATENT_SCHEMA_VERSION: str = LATENT_SCHEMA_VERSION
UCSF_PDGM_LATENT_SPATIAL: tuple[int, int, int] = LATENT_SPATIAL
UCSF_PDGM_LATENT_CHANNELS: int = LATENT_CHANNELS
UCSF_PDGM_LATENT_SEQUENCE_MAP: dict[str, str] = LATENT_SEQUENCE_MAP

# UCSF-PDGM native and padded volume shapes — cohort-specific, not in the
# neutral manifest.
UCSF_PDGM_IMAGE_NATIVE_SHAPE: tuple[int, int, int] = (240, 240, 155)
UCSF_PDGM_IMAGE_PADDED_SHAPE: tuple[int, int, int] = (192, 224, 192)

# UCSF-PDGM clinical metadata fields — 15 variables written verbatim into
# the latent H5's ``metadata/*`` group.
_UCSF_PDGM_METADATA_FIELDS: list[dict[str, str]] = [
    {"path": "metadata/sex", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Biological sex (M/F)."},
    {"path": "metadata/age", "dtype": "float32", "units": "years",
     "description": "Age at MRI acquisition."},
    {"path": "metadata/who_grade", "dtype": "int8", "units": "WHO_grade",
     "description": "WHO CNS tumour grade (1-4); -1 if unknown."},
    {"path": "metadata/diagnosis", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Final pathologic diagnosis (WHO 2021)."},
    {"path": "metadata/mgmt_status", "dtype": "vlen-str", "units": "dimensionless",
     "description": "MGMT methylation status."},
    {"path": "metadata/mgmt_index", "dtype": "vlen-str", "units": "dimensionless",
     "description": "MGMT methylation index as reported (string)."},
    {"path": "metadata/codel_1p19q", "dtype": "vlen-str", "units": "dimensionless",
     "description": "1p/19q codeletion status."},
    {"path": "metadata/idh", "dtype": "vlen-str", "units": "dimensionless",
     "description": "IDH mutation status."},
    {"path": "metadata/dead", "dtype": "int8", "units": "boolean",
     "description": "Vital status at last follow-up (1=dead, 0=alive); -1 if unknown."},
    {"path": "metadata/os_days", "dtype": "float32", "units": "days",
     "description": "Overall survival in days; NaN if unknown."},
    {"path": "metadata/eor", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Extent of resection."},
    {"path": "metadata/biopsy_prior_imaging", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Whether biopsy preceded MRI acquisition (Yes/No)."},
    {"path": "metadata/brats21_id", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Corresponding BraTS-2021 case ID; empty if not present."},
    {"path": "metadata/brats21_seg_cohort", "dtype": "vlen-str", "units": "dimensionless",
     "description": "BraTS-2021 segmentation cohort assignment."},
    {"path": "metadata/brats21_mgmt_cohort", "dtype": "vlen-str", "units": "dimensionless",
     "description": "BraTS-2021 MGMT cohort assignment."},
]

# Default manifest for the canonical UCSF-PDGM v2 run (four sequences,
# NETC/ED/ET soft mask, full clinical metadata).
UCSF_PDGM_LATENT_DEFAULT_MODALITIES: list[str] = ["t1pre", "t1c", "t2", "flair"]
UCSF_PDGM_LATENT_MANIFEST = build_latent_manifest(
    modalities=UCSF_PDGM_LATENT_DEFAULT_MODALITIES,
    mask_output_channels=3,
    cohort="UCSF-PDGM",
    metadata_fields=_UCSF_PDGM_METADATA_FIELDS,
)

__all__ = [
    "UCSF_PDGM_IMAGE_NATIVE_SHAPE",
    "UCSF_PDGM_IMAGE_PADDED_SHAPE",
    "UCSF_PDGM_LATENT_CHANNELS",
    "UCSF_PDGM_LATENT_DEFAULT_MODALITIES",
    "UCSF_PDGM_LATENT_MANIFEST",
    "UCSF_PDGM_LATENT_SCHEMA_VERSION",
    "UCSF_PDGM_LATENT_SEQUENCE_MAP",
    "UCSF_PDGM_LATENT_SPATIAL",
    "_UCSF_PDGM_METADATA_FIELDS",
    "build_latent_manifest",
]

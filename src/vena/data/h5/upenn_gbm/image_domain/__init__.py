"""UPENN-GBM image-domain H5 cache (NIfTI → H5)."""

from __future__ import annotations

from .convert import UPENNGBMImageH5Config, UPENNGBMImageH5Converter
from .manifest import (
    UPENN_GBM_IMAGE_EXPECTED_SHAPE,
    UPENN_GBM_IMAGE_MANIFEST,
    UPENN_GBM_IMAGE_SCHEMA_VERSION,
    UPENN_GBM_IMAGE_SEQUENCE_MAP,
    UPENN_GBM_LABEL_SYSTEM,
    UPENN_GBM_METADATA_FIELDS,
    MetadataFieldSpec,
)

__all__ = [
    "UPENN_GBM_IMAGE_EXPECTED_SHAPE",
    "UPENN_GBM_IMAGE_MANIFEST",
    "UPENN_GBM_IMAGE_SCHEMA_VERSION",
    "UPENN_GBM_IMAGE_SEQUENCE_MAP",
    "UPENN_GBM_LABEL_SYSTEM",
    "UPENN_GBM_METADATA_FIELDS",
    "MetadataFieldSpec",
    "UPENNGBMImageH5Config",
    "UPENNGBMImageH5Converter",
]

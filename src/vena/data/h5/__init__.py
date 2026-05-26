"""Self-describing HDF5 caches consumed by VENA training and evaluation."""

from .shared import (
    DatasetSpec,
    H5Manifest,
    H5Writer,
    assert_h5_valid,
    make_nested_cv_splits,
    validate_h5,
)
from .ucsf_pdgm.image_domain import (
    UCSF_PDGM_IMAGE_EXPECTED_SHAPE,
    UCSF_PDGM_IMAGE_MANIFEST,
    UCSF_PDGM_IMAGE_SCHEMA_VERSION,
    UCSFPDGMImageH5Config,
    UCSFPDGMImageH5Converter,
)

__all__ = [
    "UCSF_PDGM_IMAGE_EXPECTED_SHAPE",
    "UCSF_PDGM_IMAGE_MANIFEST",
    "UCSF_PDGM_IMAGE_SCHEMA_VERSION",
    "DatasetSpec",
    "H5Manifest",
    "H5Writer",
    "UCSFPDGMImageH5Config",
    "UCSFPDGMImageH5Converter",
    "assert_h5_valid",
    "make_nested_cv_splits",
    "validate_h5",
]

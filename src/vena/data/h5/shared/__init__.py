"""Shared building blocks for self-describing H5 dataset converters."""

from .crop import DEFAULT_CROP_BOX, CropGeometryError, compute_crop_origin
from .exceptions import H5ConvertError, H5Error, H5SchemaError, H5ValidationError
from .provenance import now_iso_utc, resolve_git_sha, sha256_file
from .schema import DatasetKind, DatasetSpec, DTypeTag, H5Manifest
from .splits import NestedCVSplits, make_cohort_splits, make_nested_cv_splits
from .validator import assert_h5_valid, validate_h5
from .writer import H5Writer, assign_row, open_writer

__all__ = [
    "DEFAULT_CROP_BOX",
    "DTypeTag",
    "CropGeometryError",
    "DatasetKind",
    "DatasetSpec",
    "H5ConvertError",
    "H5Error",
    "H5Manifest",
    "H5SchemaError",
    "H5ValidationError",
    "H5Writer",
    "NestedCVSplits",
    "assert_h5_valid",
    "assign_row",
    "compute_crop_origin",
    "make_cohort_splits",
    "make_nested_cv_splits",
    "now_iso_utc",
    "open_writer",
    "resolve_git_sha",
    "sha256_file",
    "validate_h5",
]

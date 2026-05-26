"""Manifest for the UCSF-PDGM image-domain H5 cache (v1.0.0).

The cache stacks the four MR sequences and the tumour segmentation across
patients into ``(N, H, W, D)`` tensors with ``chunks=(1, H, W, D)``. Cohort
metadata is preserved verbatim from ``UCSF-PDGM-metadata_v5.csv``: ``"unknown"``
strings stay as strings, ``NaN`` becomes the empty string for ``vlen-str``
fields and ``-1`` for integer fields.

Splits live under ``splits/test`` (held-out, shared across folds) and
``splits/cv/fold_{0..4}/{train,val}``, all as ``vlen-str`` of patient IDs.

The full manifest is serialised into the H5 file's ``manifest_json`` root
attribute at write time, so a consumer holding only the file can re-derive
the structure with :meth:`H5Manifest.from_json`.
"""

from __future__ import annotations

from typing import TypedDict

from vena.data.h5.shared import DatasetSpec, H5Manifest

UCSF_PDGM_IMAGE_SCHEMA_VERSION = "1.0.0"
UCSF_PDGM_IMAGE_EXPECTED_SHAPE: tuple[int, int, int] = (240, 240, 155)


class MetadataFieldSpec(TypedDict):
    """One row in :data:`UCSF_PDGM_METADATA_FIELDS`.

    Used both to (a) declare the H5 dataset (``path``, ``dtype``, ``units``,
    ``description``) and (b) pull the corresponding column from the cohort
    CSV (``csv_column``, ``cast``).
    """

    path: str
    csv_column: str
    dtype: str
    cast: str
    units: str
    description: str


# Map from manifest modality slug (lowercase, training-side names) to the
# niigz modality literal used by ``UCSFPDGMDataset.load_modality``. The
# manifest exposes ``t1pre`` (proposal vocabulary) while source files are
# named ``UCSF-PDGM-NNNN_T1_bias.nii.gz``.
UCSF_PDGM_IMAGE_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "T1_bias",
    "t1c": "T1c_bias",
    "t2": "T2_bias",
    "flair": "FLAIR_bias",
}


UCSF_PDGM_METADATA_FIELDS: list[MetadataFieldSpec] = [
    {
        "path": "metadata/sex",
        "csv_column": "Sex",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "Biological sex (M/F).",
    },
    {
        "path": "metadata/age",
        "csv_column": "Age at MRI",
        "dtype": "float32",
        "cast": "float",
        "units": "years",
        "description": "Age at MRI acquisition.",
    },
    {
        "path": "metadata/who_grade",
        "csv_column": "WHO CNS Grade",
        "dtype": "int8",
        "cast": "int",
        "units": "WHO_grade",
        "description": "WHO CNS tumour grade (1-4); -1 if unknown.",
    },
    {
        "path": "metadata/diagnosis",
        "csv_column": "Final pathologic diagnosis (WHO 2021)",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "Final pathologic diagnosis (WHO 2021).",
    },
    {
        "path": "metadata/mgmt_status",
        "csv_column": "MGMT status",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "MGMT methylation status (positive/negative/indeterminate/unknown).",
    },
    {
        "path": "metadata/mgmt_index",
        "csv_column": "MGMT index",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "MGMT methylation index as reported (string; may be 'unknown').",
    },
    {
        "path": "metadata/codel_1p19q",
        "csv_column": "1p/19q",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "1p/19q codeletion status (codeleted/not-codeleted/unknown).",
    },
    {
        "path": "metadata/idh",
        "csv_column": "IDH",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "IDH mutation status (wildtype/mutant/unknown).",
    },
    {
        "path": "metadata/dead",
        "csv_column": "1-dead 0-alive",
        "dtype": "int8",
        "cast": "int",
        "units": "boolean",
        "description": "Vital status at last follow-up (1=dead, 0=alive); -1 if unknown.",
    },
    {
        "path": "metadata/os_days",
        "csv_column": "OS",
        "dtype": "float32",
        "cast": "float",
        "units": "days",
        "description": "Overall survival in days; NaN if unknown.",
    },
    {
        "path": "metadata/eor",
        "csv_column": "EOR",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "Extent of resection (GTR/STR/biopsy/...).",
    },
    {
        "path": "metadata/biopsy_prior_imaging",
        "csv_column": "Biopsy prior to imaging",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "Whether biopsy preceded MRI acquisition (Yes/No).",
    },
    {
        "path": "metadata/brats21_id",
        "csv_column": "BraTS21 ID",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "Corresponding BraTS-2021 case ID; empty if not present.",
    },
    {
        "path": "metadata/brats21_seg_cohort",
        "csv_column": "BraTS21 Segmentation Cohort",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "BraTS-2021 segmentation cohort assignment.",
    },
    {
        "path": "metadata/brats21_mgmt_cohort",
        "csv_column": "BraTS21 MGMT Cohort",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "BraTS-2021 MGMT cohort assignment.",
    },
]


def _build_manifest() -> H5Manifest:
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Patient identifier in zero-padded form (UCSF-PDGM-NNNN).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1pre",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 pre-contrast, N4 bias-corrected; raw intensities, no rescale.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1c",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 post-contrast (gadolinium), N4 bias-corrected; raw intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t2",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T2-weighted, N4 bias-corrected; raw intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/flair",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="FLAIR, N4 bias-corrected; raw intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/tumor",
            dtype="int8",
            kind="mask",
            units="label",
            description="BraTS-style tumour labels {0=bg, 1=necrosis, 2=edema, 4=enhancing}.",
            leading_dim="n_scans",
        ),
    ]
    for entry in UCSF_PDGM_METADATA_FIELDS:
        datasets.append(
            DatasetSpec(
                path=entry["path"],
                dtype=entry["dtype"],  # type: ignore[arg-type]
                kind="metadata",
                units=entry["units"],
                description=entry["description"],
                leading_dim="n_scans",
            )
        )

    return H5Manifest(
        schema_version=UCSF_PDGM_IMAGE_SCHEMA_VERSION,
        cohort="UCSF-PDGM",
        domain="image",
        expected_shape=UCSF_PDGM_IMAGE_EXPECTED_SHAPE,
        datasets=datasets,
        splits_spec={
            "test": "splits/test  vlen-str  held-out patient IDs (shared across all folds).",
            "cv": "splits/cv/fold_{0..K-1}/{train,val}  vlen-str  per-fold patient IDs.",
        },
        extras={
            "intensity_policy": "raw N4-bias-corrected intensities; normalisation deferred to dataloader.",
            "tumor_label_set": "BraTS21: {0, 1, 2, 4}",
            "spacing_mm": "(1.0, 1.0, 1.0)",
            "space_convention": "LPS (matches MAISI-V2 expectations).",
        },
    )


UCSF_PDGM_IMAGE_MANIFEST: H5Manifest = _build_manifest()

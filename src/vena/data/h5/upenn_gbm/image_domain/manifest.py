"""Manifest for the UPENN-GBM image-domain H5 cache (v2.0.0).

UPENN-GBM (University of Pennsylvania Glioblastoma cohort, Sci Data 2022) is
cross-sectional preoperative GBM, already skull-stripped, in SRI24 1 mm iso
``(240, 240, 155)``. Tumour segmentations follow the BraTS-2021 label system
``{0, 1, 2, 4}`` (the manual ones are gold-standard; the automated ones come
from the released ``automated_segm`` collection — the reader prefers manual
where available).

The cache stacks the four MR sequences and the tumour segmentation across
patients into ``(N, 240, 240, 155)`` tensors with ``chunks=(1, 240, 240, 155)``
and gzip-4 compression. The single non-trivial metadata field is
``metadata/brats21_id``, populated from a side lookup CSV
(``UPENN-GBM_brats21_lookup_v1.csv``); this drives the cross-cohort dedup
preflight via ``bridge_fields: {UPENN-GBM: metadata/brats21_id}`` exactly as
UCSF-PDGM does.

Splits live under ``splits/test`` (held-out, shared across folds) and
``splits/cv/fold_{0..4}/{train,val}``, all as ``vlen-str`` of patient IDs.

The full manifest is serialised into the H5 file's ``manifest_json`` root
attribute at write time.
"""

from __future__ import annotations

from typing import TypedDict

from vena.data.h5.shared import DatasetSpec, H5Manifest

UPENN_GBM_IMAGE_SCHEMA_VERSION = "2.0.0"
# Native LPS shape, preserved verbatim. The common crop box (192,224,192) is
# applied at encode time, not stored here; see crop/origin and crop_box attr.
UPENN_GBM_IMAGE_EXPECTED_SHAPE: tuple[int, int, int] = (240, 240, 155)
UPENN_GBM_LABEL_SYSTEM = "BraTS2021"  # tumour labels {0, 1, 2, 4}


class MetadataFieldSpec(TypedDict):
    """One row in :data:`UPENN_GBM_METADATA_FIELDS`.

    Mirrors :class:`vena.data.h5.ucsf_pdgm.image_domain.manifest.MetadataFieldSpec`.
    """

    path: str
    csv_column: str
    dtype: str
    cast: str
    units: str
    description: str


# H5 modality slug → BraTS file suffix used by the UPENN-GBM release.
UPENN_GBM_IMAGE_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "T1",
    "t1c": "T1GD",
    "t2": "T2",
    "flair": "FLAIR",
}


UPENN_GBM_METADATA_FIELDS: list[MetadataFieldSpec] = [
    {
        "path": "metadata/brats21_id",
        "csv_column": "brats21_id",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": "Corresponding BraTS-2021 case ID; empty if not present.",
    },
    {
        "path": "metadata/brats21_data_collection",
        "csv_column": "brats21_data_collection",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": (
            "BraTS-2021 Data Collection label "
            "({UPENN-GBM, UPENN-GBM_Additional}); empty if not present."
        ),
    },
    {
        "path": "metadata/seg_source",
        "csv_column": "seg_source",
        "dtype": "vlen-str",
        "cast": "str",
        "units": "dimensionless",
        "description": (
            "Origin of the tumour segmentation for this patient: 'manual' if a row "
            "from images_segm/ was used, 'automated' if automated_segm/ was used."
        ),
    },
]


def _build_manifest() -> H5Manifest:
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Patient identifier (UPENN-GBM-NNNNN_NN).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1pre",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 pre-contrast; raw intensities, no rescale.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1c",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 post-contrast (gadolinium); raw intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t2",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T2-weighted; raw intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/flair",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="FLAIR; raw intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/tumor",
            dtype="int8",
            kind="mask",
            units="label",
            description="BraTS-2021 tumour labels {0=bg, 1=necrosis, 2=edema, 4=enhancing}.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/brain",
            dtype="int8",
            kind="mask",
            units="label",
            description="Binary brain mask (1=brain, 0=background) — union of nonzero across modalities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="crop/origin",
            dtype="int32",
            kind="metadata",
            units="voxels",
            description=(
                "Per-scan brain-centred start index (H,W,D) in the native LPS grid "
                "of the common crop box (see root attr crop_box); may be negative."
            ),
            leading_dim="n_scans",
        ),
        # CSR patient grouping (trivial 1:1 for cross-sectional UPENN-GBM).
        DatasetSpec(
            path="patients/offsets",
            dtype="int32",
            kind="metadata",
            units="dimensionless",
            description="CSR offsets, length n_patients+1.",
            leading_dim=None,
        ),
        DatasetSpec(
            path="patients/keys",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Unique patient keys, length n_patients, in offset order.",
            leading_dim=None,
        ),
    ]
    for entry in UPENN_GBM_METADATA_FIELDS:
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
        schema_version=UPENN_GBM_IMAGE_SCHEMA_VERSION,
        cohort="UPENN-GBM",
        domain="image",
        expected_shape=UPENN_GBM_IMAGE_EXPECTED_SHAPE,
        datasets=datasets,
        splits_spec={
            "test": "splits/test  vlen-str  held-out patient IDs (shared across all folds).",
            "cv": "splits/cv/fold_{0..K-1}/{train,val}  vlen-str  per-fold patient IDs.",
        },
        extras={
            "intensity_policy": "raw intensities; brain-only percentile norm at encode time.",
            "tumor_label_set": "BraTS-2021: {0, 1, 2, 4}",
            "spacing_mm": "(1.0, 1.0, 1.0)",
            "space_convention": "LPS (SRI24 atlas; preserved verbatim).",
            "crop_box_hwd": "(192, 224, 192)",
            "seg_policy": "manual seg (images_segm/) preferred; falls back to automated_segm/.",
        },
    )


UPENN_GBM_IMAGE_MANIFEST: H5Manifest = _build_manifest()

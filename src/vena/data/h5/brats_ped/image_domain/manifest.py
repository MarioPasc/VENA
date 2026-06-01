"""Manifest builder for the BraTS-PED 2024 image-domain H5 cache.

The cohort is OOD-test-only: ``splits/test`` covers every patient; there are
no CV folds. Tumour labels follow the BraTS-2023 convention ``{0, 1, 2, 3}``;
the encode-time remap to BraTS-2021 ``{0, 1, 2, 4}`` is applied downstream
when ``label_system == "BraTS2023"`` is set.

Input intensities are native-scanner (HD-BET-stripped: brain only, raw scale);
percentile normalisation happens at encode time via MAISI.
"""

from __future__ import annotations

from vena.data.h5.shared import DatasetSpec, H5Manifest

BRATS_PED_IMAGE_SCHEMA_VERSION = "2.0.0"
BRATS_PED_IMAGE_EXPECTED_SHAPE: tuple[int, int, int] = (240, 240, 155)
BRATS_PED_LABEL_SYSTEM = "BraTS2023"  # tumour labels {0, 1, 2, 3}

# H5 modality slug → BraTS file suffix used by the source NIfTI names.
BRATS_PED_IMAGE_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "t1n",
    "t1c": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}


def build_brats_ped_image_manifest(cohort_tag: str = "BraTS-PED") -> H5Manifest:
    """Build the image-domain manifest for the BraTS-PED 2024 cohort."""
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Patient identifier (BraTS-PED-NNNNN-NNN; one row per patient).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1pre",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 pre-contrast (t1n); HD-BET skull-stripped, SRI24 1 mm iso, native scanner intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1c",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 post-contrast (t1c); HD-BET skull-stripped, SRI24 1 mm iso, native scanner intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t2",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T2-weighted (t2w); HD-BET skull-stripped, SRI24 1 mm iso, native scanner intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/flair",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="FLAIR (t2f); HD-BET skull-stripped, SRI24 1 mm iso, native scanner intensities.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/tumor",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "BraTS-2023 tumour labels {0=bg, 1=NCR, 2=ED, 3=ET}. The "
                "encode-time remap to BraTS-2021 {0,1,2,4} is applied "
                "downstream when label_system='BraTS2023' is set."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/brain",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "Binary brain mask (1=brain, 0=background) derived as the "
                "union of nonzero voxels across {t1pre, t1c, t2, flair} after "
                "HD-BET skull-strip (background is exactly zero by construction)."
            ),
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
        DatasetSpec(
            path="patients/offsets",
            dtype="int32",
            kind="metadata",
            units="dimensionless",
            description=(
                "CSR offsets, length n_patients+1; scans of patient k are rows "
                "[offsets[k]:offsets[k+1]] (1:1 cross-sectional)."
            ),
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

    return H5Manifest(
        schema_version=BRATS_PED_IMAGE_SCHEMA_VERSION,
        cohort=cohort_tag,
        domain="image",
        expected_shape=BRATS_PED_IMAGE_EXPECTED_SHAPE,
        datasets=datasets,
        splits_spec={
            "test": "splits/test  vlen-str  every patient (OOD-test only, no CV folds).",
        },
        extras={
            "intensity_policy": "Native scanner intensities post HD-BET skull-strip; brain-only percentile norm at encode time.",
            "tumor_label_set": "BraTS-2023: {0, 1, 2, 3}",
            "spacing_mm": "(1.0, 1.0, 1.0)",
            "space_convention": "LPS (BraTS-2024-PED native, SRI24).",
            "crop_box_hwd": "(192, 224, 192)",
            "split_strategy": "all patients in splits/test (OOD-test only).",
            "skull_strip_tool": "HD-BET v2 (BraTS-PED is defaced only; skull was stripped via routines/preprocess/brats_ped_skullstrip).",
            "release": "BraTS-PED 2024 Challenge — Training set (pediatric HGG).",
        },
    )

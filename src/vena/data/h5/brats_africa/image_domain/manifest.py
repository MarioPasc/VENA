"""Manifest builder for BraTS-Africa image-domain H5 caches.

The same converter writes two artifacts (``BraTS_Africa_glioma_image.h5`` and
``BraTS_Africa_other_image.h5``); the cohort tag in the manifest differentiates
them. Both share the SRI24 atlas grid ``(240, 240, 155)``, BraTS-2023 tumour
labels ``{0, 1, 2, 3}``, and the OOD test-only split layout (``splits/test``
covering every patient; no CV folds).
"""

from __future__ import annotations

from vena.data.h5.shared import DatasetSpec, H5Manifest

BRATS_AFRICA_IMAGE_SCHEMA_VERSION = "2.0.0"
BRATS_AFRICA_IMAGE_EXPECTED_SHAPE: tuple[int, int, int] = (240, 240, 155)
BRATS_AFRICA_LABEL_SYSTEM = "BraTS2023"  # tumour labels {0, 1, 2, 3}

# H5 modality slug → BraTS file suffix used by the source NIfTI names.
BRATS_AFRICA_IMAGE_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "t1n",
    "t1c": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}


def build_brats_africa_image_manifest(cohort_tag: str, subset_label: str) -> H5Manifest:
    """Build an image-domain manifest for one of the BraTS-Africa subsets.

    Parameters
    ----------
    cohort_tag
        Human-readable cohort name written into the H5 ``cohort`` root attr
        (e.g. ``"BraTS-Africa-Glioma"``).
    subset_label
        Source subdirectory tag echoed into ``extras`` for traceability
        (e.g. ``"95_Glioma"`` or ``"51_OtherNeoplasms"``).
    """
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Patient identifier (BraTS-SSA-NNNNN-000; one row per patient).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1pre",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 pre-contrast (t1n); z-score-normalised intra-brain, skull-stripped, SRI24 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1c",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 post-contrast (t1c); z-score-normalised intra-brain, skull-stripped, SRI24 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t2",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T2-weighted (t2w); z-score-normalised intra-brain, skull-stripped, SRI24 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/flair",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="FLAIR (t2f); z-score-normalised intra-brain, skull-stripped, SRI24 1 mm iso.",
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
                "voxels with any non-zero intensity across {t1pre, t1c, t2, "
                "flair} (z-score data has negative tail; zero indicates "
                "background per the BraTS pipeline)."
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
        schema_version=BRATS_AFRICA_IMAGE_SCHEMA_VERSION,
        cohort=cohort_tag,
        domain="image",
        expected_shape=BRATS_AFRICA_IMAGE_EXPECTED_SHAPE,
        datasets=datasets,
        splits_spec={
            "test": "splits/test  vlen-str  every patient (OOD-test only, no CV folds).",
        },
        extras={
            "intensity_policy": "BraTS-pipeline intra-brain z-score; brain-only percentile norm at encode time.",
            "tumor_label_set": "BraTS-2023: {0, 1, 2, 3}",
            "spacing_mm": "(1.0, 1.0, 1.0)",
            "space_convention": "LPS (reoriented at write time from BraTS-SSA native).",
            "crop_box_hwd": "(192, 224, 192)",
            "subset_label": subset_label,
            "split_strategy": "all patients in splits/test (OOD-test only).",
        },
    )

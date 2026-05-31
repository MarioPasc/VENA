"""Manifest for the LUMIERE image-domain H5 cache (schema 2.0.0).

Longitudinal cohort: each ``(patient, session)`` is one row; CSR
``patients/{offsets, keys}`` keeps a patient's sessions contiguous. Source
volumes come from ``DeepBraTumIA-segmentation/atlas/skull_strip/`` and the
companion ``atlas/segmentation/seg_mask.nii.gz`` — MNI152 1 mm isotropic,
shape ``(182, 218, 182)``, skull-stripped. Tumour labels follow the BraTS-2023
convention ``{0, 1, 2, 3}``; the encode-time remap to BraTS-2021 ``{0,1,2,4}``
is applied downstream when ``label_system='BraTS2023'``.

Splits live under ``splits/test`` (10 held-out patients) and
``splits/cv/fold_{0..4}/{train,val}`` (5-fold CV on the remaining 81),
written as ``vlen-str`` patient IDs. Patient-level splits prevent
session-level leakage.
"""

from __future__ import annotations

from vena.data.h5.shared import DatasetSpec, H5Manifest

LUMIERE_IMAGE_SCHEMA_VERSION = "2.0.0"
LUMIERE_IMAGE_EXPECTED_SHAPE: tuple[int, int, int] = (182, 218, 182)
LUMIERE_LABEL_SYSTEM = "BraTS2023"


def _build_manifest() -> H5Manifest:
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Session identifier (Patient-NNN__week-NNN[-N]); one row per session.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1pre",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 pre-contrast (t1_skull_strip.nii.gz); MNI152 1 mm iso, skull-stripped.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1c",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 post-contrast (ct1_skull_strip.nii.gz); MNI152 1 mm iso, skull-stripped.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t2",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T2-weighted (t2_skull_strip.nii.gz); MNI152 1 mm iso, skull-stripped.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/flair",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="FLAIR (flair_skull_strip.nii.gz); MNI152 1 mm iso, skull-stripped.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/tumor",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "DeepBraTumIA atlas-space tumour labels (BraTS-2023 convention "
                "{0=bg, 1=NCR, 2=ED, 3=ET}); seg_mask.nii.gz under "
                "DeepBraTumIA-segmentation/atlas/segmentation/."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/brain",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "Binary brain mask (1=brain, 0=background) from "
                "DeepBraTumIA-segmentation/atlas/skull_strip/brain_mask.nii.gz."
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
        # CSR patient grouping — lengths are n_patients(+1), not n_scans.
        DatasetSpec(
            path="patients/offsets",
            dtype="int32",
            kind="metadata",
            units="dimensionless",
            description=(
                "CSR offsets, length n_patients+1; sessions of patient k are "
                "rows [offsets[k]:offsets[k+1]]."
            ),
            leading_dim=None,
        ),
        DatasetSpec(
            path="patients/keys",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Unique patient keys (Patient-NNN), length n_patients, in offset order.",
            leading_dim=None,
        ),
        # Per-session metadata derived from the session directory name.
        DatasetSpec(
            path="metadata/patient_id",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Patient identifier for this row's session (Patient-NNN).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="metadata/week",
            dtype="int32",
            kind="metadata",
            units="weeks",
            description="Week index (NNN) parsed from the session directory name.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="metadata/week_repeat",
            dtype="int32",
            kind="metadata",
            units="dimensionless",
            description="Repeat suffix on the week-NNN-K session name; -1 when absent.",
            leading_dim="n_scans",
        ),
    ]

    return H5Manifest(
        schema_version=LUMIERE_IMAGE_SCHEMA_VERSION,
        cohort="LUMIERE",
        domain="image",
        expected_shape=LUMIERE_IMAGE_EXPECTED_SHAPE,
        datasets=datasets,
        splits_spec={
            "test": "splits/test  vlen-str  held-out patient IDs (shared across folds).",
            "cv": "splits/cv/fold_{0..K-1}/{train,val}  vlen-str  per-fold patient IDs.",
        },
        extras={
            "intensity_policy": "raw skull-stripped intensities from DeepBraTumIA atlas; brain-only percentile norm at encode time.",
            "tumor_label_set": "BraTS-2023: {0, 1, 2, 3} (DeepBraTumIA atlas seg).",
            "spacing_mm": "(1.0, 1.0, 1.0)",
            "space_convention": "LPS (DeepBraTumIA atlas; reorientation may be a no-op or single flip).",
            "crop_box_hwd": "(192, 224, 192) (matches BraTS-GLI 182x218x182 cohort).",
            "longitudinal_note": (
                "LUMIERE is longitudinal: each (Patient, week) pair is one row. "
                "CSR patients/offsets + patients/keys encodes the grouping. Splits "
                "are patient-level to prevent leakage across timepoints."
            ),
        },
    )


LUMIERE_IMAGE_MANIFEST: H5Manifest = _build_manifest()

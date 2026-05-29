"""Manifest for the BraTS-GLI image-domain H5 cache (v2.0.0).

The cache stacks the four MR sequences (t1pre/t1c/t2/flair) and the tumour
segmentation across sessions into ``(N, 182, 218, 182)`` tensors with
``chunks=(1, 182, 218, 182)`` and gzip-4 compression. Sessions are ordered so
that all sessions of a patient are contiguous, enabling CSR-style access via
``patients/offsets`` and ``patients/keys``.

Unlike UCSF-PDGM, this cohort carries no metadata CSV. Splits are computed at
the patient level (nested CV, role="cv") and stored under ``splits/test`` and
``splits/cv/fold_{0..K-1}/{train,val}`` as vlen-str of patient IDs.

The full manifest is serialised into the H5 file's ``manifest_json`` root
attribute at write time.
"""

from __future__ import annotations

from vena.data.h5.shared import DatasetSpec, H5Manifest

BRATS_GLI_IMAGE_SCHEMA_VERSION = "2.0.0"

# Native skull-stripped shape at 1 mm isotropic, LAS → reoriented to LPS at
# write time (axis flip on the second axis: LAS→LPS = sign-flip of P axis).
# Shape is identical pre- and post-reorientation for a flip-only transform.
BRATS_GLI_IMAGE_EXPECTED_SHAPE: tuple[int, int, int] = (182, 218, 182)

BRATS_GLI_LABEL_SYSTEM = "BraTS2023"  # tumour labels {0, 1, 2, 3}

# Map from H5 modality slug to the BraTS file suffix used in filenames.
# E.g. session ``BraTS-GLI-00001-000`` → ``BraTS-GLI-00001-000-t1n.nii.gz``.
BRATS_GLI_IMAGE_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "t1n",
    "t1c": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}


def _build_manifest() -> H5Manifest:
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Session identifier (BraTS-GLI-PPPPP-TTT).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1pre",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 pre-contrast (t1n); raw intensities, skull-stripped, 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1c",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 post-contrast (t1c); raw intensities, skull-stripped, 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t2",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T2-weighted (t2w); raw intensities, skull-stripped, 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/flair",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="FLAIR (t2f); raw intensities, skull-stripped, 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/tumor",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "BraTS2023 tumour labels {0=bg, 1=necrotic core, 2=peritumoral "
                "oedema, 3=enhancing tumour}. Label 4 from older BraTS releases "
                "was remapped to 3 in the 2023 challenge."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/brain",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "Binary brain mask (1=brain, 0=background) derived as nonzero "
                "foreground of t1n after LPS reorientation (skull-stripped input)."
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
        # CSR patient grouping: lengths are n_patients(+1), not n_scans, so
        # leading_dim is None (validator skips the n_scans equality check).
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
            description="Unique patient keys (BraTS-GLI-PPPPP), length n_patients, in offset order.",
            leading_dim=None,
        ),
    ]

    return H5Manifest(
        schema_version=BRATS_GLI_IMAGE_SCHEMA_VERSION,
        cohort="BraTS-GLI",
        domain="image",
        expected_shape=BRATS_GLI_IMAGE_EXPECTED_SHAPE,
        datasets=datasets,
        splits_spec={
            "test": "splits/test  vlen-str  held-out patient IDs (shared across all folds).",
            "cv": "splits/cv/fold_{0..K-1}/{train,val}  vlen-str  per-fold patient IDs.",
        },
        extras={
            "intensity_policy": "raw intensities; brain-only percentile norm at encode time.",
            "tumor_label_set": "BraTS2023: {0, 1, 2, 3}",
            "spacing_mm": "(1.0, 1.0, 1.0)",
            "space_convention": "LPS (reoriented from LAS at write time via axis flip).",
            "crop_box_hwd": "(192, 224, 192)",
            "longitudinal_note": (
                "BraTS-GLI is a longitudinal cohort: some patients have >1 session. "
                "CSR patients/offsets + patients/keys encodes the grouping. Splits are "
                "patient-level to prevent data leakage across timepoints."
            ),
        },
    )


BRATS_GLI_IMAGE_MANIFEST: H5Manifest = _build_manifest()

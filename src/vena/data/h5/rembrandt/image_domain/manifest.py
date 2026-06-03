"""Manifest for the REMBRANDT image-domain H5 cache (schema 2.0.0).

Cross-sectional cohort: one scan per patient, native SRI24 shape
``(240, 240, 155)`` at 1 mm isotropic, LPS. **HD-BET skull-stripped** by the
upstream routine; tumour labels follow the BraTS-2021 convention
``{0, 1, 2, 4}`` (GlistrBoost CBICA pipeline). Splits follow the canonical
multi-cohort layout: ``splits/test`` (held-out) +
``splits/cv/fold_0/{train, val}`` (single fold of the remaining patients).
N=63 is too small for stable nested K-fold CV, so a single fold (53/5/5
train/val/test, seed 42) is used.
"""

from __future__ import annotations

from vena.data.h5.shared import DatasetSpec, H5Manifest

REMBRANDT_IMAGE_SCHEMA_VERSION = "2.0.0"
REMBRANDT_IMAGE_EXPECTED_SHAPE: tuple[int, int, int] = (240, 240, 155)
REMBRANDT_LABEL_SYSTEM = "BraTS2021"  # GlistrBoost tumour labels {0, 1, 2, 4}

# H5 modality slug → REMBRANDT filename infix (between the date and the
# trailing ``_LPS_rSRI.nii.gz``).
REMBRANDT_IMAGE_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "t1",
    "t1c": "t1ce",
    "t2": "t2",
    "flair": "flair",
}


def build_rembrandt_image_manifest(cohort_tag: str = "REMBRANDT") -> H5Manifest:
    """Build the image-domain manifest for the REMBRANDT cohort."""
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description=(
                "Patient/session identifier (<pid>_<YYYY.MM.DD>; one row per "
                "patient — REMBRANDT is cross-sectional)."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1pre",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description=(
                "T1 pre-contrast (REMBRANDT filename suffix '_t1_'); HD-BET "
                "skull-stripped, SRI24 1 mm iso, native scanner intensities."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1c",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description=(
                "T1 post-contrast (REMBRANDT filename suffix '_t1ce_'); HD-BET "
                "skull-stripped, SRI24 1 mm iso, native scanner intensities."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t2",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description=(
                "T2-weighted (REMBRANDT filename suffix '_t2_'); HD-BET "
                "skull-stripped, SRI24 1 mm iso, native scanner intensities."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/flair",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description=(
                "FLAIR (REMBRANDT filename suffix '_flair_'); HD-BET "
                "skull-stripped, SRI24 1 mm iso, native scanner intensities."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/tumor",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "BraTS-2021 tumour labels {0=bg, 1=NCR/NET, 2=ED, 4=ET} from the "
                "GlistrBoost CBICA pipeline."
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
                "[offsets[k]:offsets[k+1]] (1:1 for REMBRANDT cross-sectional)."
            ),
            leading_dim=None,
        ),
        DatasetSpec(
            path="patients/keys",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Unique patient keys (<pid>_<date>) in offset order.",
            leading_dim=None,
        ),
    ]

    return H5Manifest(
        schema_version=REMBRANDT_IMAGE_SCHEMA_VERSION,
        cohort=cohort_tag,
        domain="image",
        expected_shape=REMBRANDT_IMAGE_EXPECTED_SHAPE,
        datasets=datasets,
        splits_spec={
            "test": "splits/test               vlen-str  held-out test patient IDs.",
            "cv": "splits/cv/fold_0/{train,val}  vlen-str  single-fold CV partition.",
        },
        extras={
            "intensity_policy": "Native scanner intensities post HD-BET skull-strip; brain-only percentile norm at encode time.",
            "tumor_label_set": "BraTS-2021: {0, 1, 2, 4}",
            "spacing_mm": "(1.0, 1.0, 1.0)",
            "space_convention": "LPS (REMBRANDT CBICA-preprocessed, SRI24).",
            "crop_box_hwd": "(192, 224, 192)",
            "split_strategy": "single random 53/5/5 train/val/test (N=63 too small for nested CV; mirrors IvyGAP).",
            "skull_strip_tool": "HD-BET v2 (REMBRANDT is registered but not stripped; skull was stripped via routines/preprocess/rembrandt_skullstrip).",
            "release": "REMBRANDT (TCIA; CBICA-preprocessed: LPS/SRI24 1 mm iso, GlistrBoost tumour seg).",
        },
    )


REMBRANDT_IMAGE_MANIFEST: H5Manifest = build_rembrandt_image_manifest()

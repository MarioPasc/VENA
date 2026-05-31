"""Manifest for the IvyGAP image-domain H5 cache (schema 2.0.0).

Cross-sectional cohort: one scan per patient, native SRI24 shape
``(240, 240, 155)`` at 1 mm isotropic, LPS, skull-stripped. Tumour labels
follow the BraTS-2021 convention ``{0, 1, 2, 4}`` (UPenn annotation; CWRU
path is preserved in ``metadata/cwru_seg_path`` but not used as the canonical
mask).

Splits live under ``splits/{train, val, test}`` as ``vlen-str`` lists of
patient IDs. The cohort is too small (34 patients) for nested K-fold CV;
the single 24/5/5 split is intentional.
"""

from __future__ import annotations

from vena.data.h5.shared import DatasetSpec, H5Manifest

IVY_GAP_IMAGE_SCHEMA_VERSION = "2.0.0"
IVY_GAP_IMAGE_EXPECTED_SHAPE: tuple[int, int, int] = (240, 240, 155)
IVY_GAP_LABEL_SYSTEM = "BraTS2021"  # tumour labels {0, 1, 2, 4}

# H5 modality slug → reader/source-filename infix. Used by the converter to
# select the right NIfTI for each row.
IVY_GAP_IMAGE_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "t1",
    "t1c": "t1gd",
    "t2": "t2",
    "flair": "flair",
}


def _build_manifest() -> H5Manifest:
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Patient identifier (W<N>; one row per patient).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1pre",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 pre-contrast (_t1_); raw intensities, skull-stripped, SRI24 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t1c",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T1 post-contrast (_t1gd_); raw intensities, skull-stripped, SRI24 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/t2",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="T2-weighted (_t2_); raw intensities, skull-stripped, SRI24 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="images/flair",
            dtype="float32",
            kind="image",
            units="intensity_au",
            description="FLAIR (_flair_); raw intensities, skull-stripped, SRI24 1 mm iso.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="masks/tumor",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "BraTS-2021 tumour labels {0=bg, 1=NCR/NET, 2=ED, 4=ET} from the "
                "UPenn rater (34/34 patients)."
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
                "foreground of t1pre after LPS load (skull-stripped input)."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="crop/origin",
            dtype="int32",
            kind="metadata",
            units="voxels",
            description=(
                "Per-scan brain-centred start index (H, W, D) in the native LPS grid "
                "of the common crop box (see root attr crop_box); may be negative."
            ),
            leading_dim="n_scans",
        ),
        # CSR patient grouping is trivial 1:1 here (cross-sectional) but stored
        # for layout uniformity with the longitudinal cohorts.
        DatasetSpec(
            path="patients/offsets",
            dtype="int32",
            kind="metadata",
            units="dimensionless",
            description=(
                "CSR offsets, length n_patients+1; scans of patient k are rows "
                "[offsets[k]:offsets[k+1]] (1:1 here)."
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
        # Per-row metadata. IvyGAP has no clinical CSV on disk; the only fields
        # we can populate are derived from the source filenames.
        DatasetSpec(
            path="metadata/scan_date",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Scan date YYYY.MM.DD parsed from the session subdirectory name.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="metadata/tumor_seg_source",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Annotation source for masks/tumor (always 'upenn' in v0.1).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="metadata/source_basename_t1pre",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Source NIfTI basename for t1pre; records the registration / bias-correction variant.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="metadata/source_basename_t1c",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Source NIfTI basename for t1c; records the registration / bias-correction variant.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="metadata/source_basename_t2",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Source NIfTI basename for t2; records the registration / bias-correction variant.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="metadata/source_basename_flair",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Source NIfTI basename for flair; records the registration / bias-correction variant.",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="metadata/cwru_seg_path",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Absolute path to the CWRU annotation if present; empty string otherwise.",
            leading_dim="n_scans",
        ),
    ]

    return H5Manifest(
        schema_version=IVY_GAP_IMAGE_SCHEMA_VERSION,
        cohort="IvyGAP",
        domain="image",
        expected_shape=IVY_GAP_IMAGE_EXPECTED_SHAPE,
        datasets=datasets,
        splits_spec={
            "train": "splits/train  vlen-str  training patient IDs.",
            "val": "splits/val    vlen-str  validation patient IDs.",
            "test": "splits/test   vlen-str  held-out test patient IDs.",
        },
        extras={
            "intensity_policy": "raw scanner intensities; brain-only percentile norm at encode time.",
            "tumor_label_set": "BraTS-2021: {0, 1, 2, 4}",
            "spacing_mm": "(1.0, 1.0, 1.0)",
            "space_convention": "LPS (native SRI24); reorient is a no-op for this cohort.",
            "crop_box_hwd": "(192, 224, 192)",
            "split_strategy": "single random 24/5/5 train/val/test (N=34 too small for nested CV).",
            "variant_precedence": "filename variant precedence: _N4_r_SS > _r3_SS > _r_SS.",
        },
    )


IVY_GAP_IMAGE_MANIFEST: H5Manifest = _build_manifest()

"""Manifest for the UCSF-PDGM latent-domain H5 cache (v0.1.0).

The cache mirrors the image-domain layout (one row per patient, stacked
``chunks=(1, …)`` datasets, ``vlen-str`` IDs and splits) but stores MAISI-V2
VAE-GAN latents instead of raw intensities, plus a MAISI-space tumour mask.

Dimensions are fixed by the spatial compression factor of the autoencoder
(``4×`` along every axis) and the post-pad image shape (``240, 240, 160``):

* image space (after depth-pad): ``(240, 240, 160)``;
* latent space:                  ``(60, 60, 40)`` with ``C = 4``.

Modalities are listed *dynamically* by the converter: the manifest is built
at write time with the subset declared in the routine config so a v1 run
encoding only ``{t1pre, t1c, t2, flair}`` does not block a later v2 run
that adds ``adc`` and ``swi`` on the same schema version. The set of
modalities actually encoded is also serialised into the ``modalities_encoded_json``
root attribute.

Tumour mask: stored at ``masks/tumor_latent`` as a multi-channel float32
soft map; channel ordering and per-class semantics are documented by the
producing downsampler via its ``to_attrs()`` dict (written verbatim into
the dataset's HDF5 attrs).

The ``priors/`` group is reserved for future per-prior arrays (vessel,
cellularity, perfusion, susceptibility). It is created as an empty group
on every run; downstream routines append datasets and stamp their own
provenance attrs (producing routine, sha256 of the source priors module,
etc.).
"""

from __future__ import annotations

from typing import TypedDict

from vena.data.h5.shared import DatasetSpec, H5Manifest

UCSF_PDGM_LATENT_SCHEMA_VERSION = "0.1.0"

# Image-domain shape after end-pad to a multiple of 8 along depth; the
# latent_domain manifest never re-validates this directly — it lives here
# to keep the magic numbers in one place.
UCSF_PDGM_IMAGE_NATIVE_SHAPE: tuple[int, int, int] = (240, 240, 155)
UCSF_PDGM_IMAGE_PADDED_SHAPE: tuple[int, int, int] = (240, 240, 160)
UCSF_PDGM_LATENT_SPATIAL: tuple[int, int, int] = (60, 60, 40)
UCSF_PDGM_LATENT_CHANNELS: int = 4

# Same MAISI integer codes table that lives in
# src/vena/model/autoencoder/maisi/configs/modality_mapping.json. Mirrored
# here so the manifest stays free of cross-package file reads at import.
UCSF_PDGM_LATENT_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "mri_t1",
    "t1c": "mri_t1c",
    "t2": "mri_t2",
    "flair": "mri_flair",
    "adc": "mri_adc",
    "swi": "mri_swi",
}

# Metadata columns are copied verbatim from the image H5. Keeping the list
# here means the latent file is *self-contained* (no need to JOIN against
# the image H5 to read who_grade for a downstream split-by-grade query).
_LATENT_METADATA_FIELDS: list[dict[str, str]] = [
    {"path": "metadata/sex", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Biological sex (M/F)."},
    {"path": "metadata/age", "dtype": "float32", "units": "years",
     "description": "Age at MRI acquisition."},
    {"path": "metadata/who_grade", "dtype": "int8", "units": "WHO_grade",
     "description": "WHO CNS tumour grade (1-4); -1 if unknown."},
    {"path": "metadata/diagnosis", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Final pathologic diagnosis (WHO 2021)."},
    {"path": "metadata/mgmt_status", "dtype": "vlen-str", "units": "dimensionless",
     "description": "MGMT methylation status."},
    {"path": "metadata/mgmt_index", "dtype": "vlen-str", "units": "dimensionless",
     "description": "MGMT methylation index as reported (string)."},
    {"path": "metadata/codel_1p19q", "dtype": "vlen-str", "units": "dimensionless",
     "description": "1p/19q codeletion status."},
    {"path": "metadata/idh", "dtype": "vlen-str", "units": "dimensionless",
     "description": "IDH mutation status."},
    {"path": "metadata/dead", "dtype": "int8", "units": "boolean",
     "description": "Vital status at last follow-up (1=dead, 0=alive); -1 if unknown."},
    {"path": "metadata/os_days", "dtype": "float32", "units": "days",
     "description": "Overall survival in days; NaN if unknown."},
    {"path": "metadata/eor", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Extent of resection."},
    {"path": "metadata/biopsy_prior_imaging", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Whether biopsy preceded MRI acquisition (Yes/No)."},
    {"path": "metadata/brats21_id", "dtype": "vlen-str", "units": "dimensionless",
     "description": "Corresponding BraTS-2021 case ID; empty if not present."},
    {"path": "metadata/brats21_seg_cohort", "dtype": "vlen-str", "units": "dimensionless",
     "description": "BraTS-2021 segmentation cohort assignment."},
    {"path": "metadata/brats21_mgmt_cohort", "dtype": "vlen-str", "units": "dimensionless",
     "description": "BraTS-2021 MGMT cohort assignment."},
]


class _ModalitySpec(TypedDict):
    slug: str
    description: str


def _build_manifest(
    modalities: list[str],
    mask_output_channels: int,
) -> H5Manifest:
    """Build the manifest for one encoder run.

    Parameters
    ----------
    modalities : list[str]
        Subset of :data:`UCSF_PDGM_LATENT_SEQUENCE_MAP` keys to allocate
        latent datasets for.
    mask_output_channels : int
        Number of channels produced by the mask downsampler. Determines the
        spatial shape declared for ``masks/tumor_latent``.
    """
    unknown = [m for m in modalities if m not in UCSF_PDGM_LATENT_SEQUENCE_MAP]
    if unknown:
        raise ValueError(
            f"unknown modalities {unknown}; available: {sorted(UCSF_PDGM_LATENT_SEQUENCE_MAP)}"
        )
    if not modalities:
        raise ValueError("at least one modality must be requested")

    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Patient identifier (UCSF-PDGM-NNNN).",
            leading_dim="n_scans",
        ),
    ]
    for slug in modalities:
        mapped = UCSF_PDGM_LATENT_SEQUENCE_MAP[slug]
        datasets.append(
            DatasetSpec(
                path=f"latents/{slug}",
                dtype="float32",
                kind="image",  # spatial array; reuse the "image" kind for shape validation
                units="latent_au",
                description=(
                    f"MAISI-V2 VAE-GAN latent of {slug} (MAISI modality slug {mapped!r}); "
                    "encoded from depth-padded image space."
                ),
                leading_dim="n_scans",
            )
        )

    datasets.append(
        DatasetSpec(
            path="masks/tumor_latent",
            dtype="float32",
            kind="mask",
            units="dimensionless",
            description=(
                "Soft tumour-label map in MAISI latent space; per-class avg-pool of the "
                "BraTS labels by default (channels = NETC, ED, ET)."
            ),
            leading_dim="n_scans",
        )
    )

    for field in _LATENT_METADATA_FIELDS:
        datasets.append(
            DatasetSpec(
                path=field["path"],
                dtype=field["dtype"],  # type: ignore[arg-type]
                kind="metadata",
                units=field["units"],
                description=field["description"],
                leading_dim="n_scans",
            )
        )

    # ``expected_shape`` would be the latent spatial shape *plus* the leading
    # channel axis, but H5Manifest.expected_shape applies only to image/mask
    # kinds and assumes shape == (N, *expected). For multi-channel latents
    # we override per-dataset via ``leading_dim`` and skip the shared
    # spatial check by setting ``expected_shape=None`` — the converter
    # asserts the exact (C, h, w, d) shape itself at create_stacked time.
    return H5Manifest(
        schema_version=UCSF_PDGM_LATENT_SCHEMA_VERSION,
        cohort="UCSF-PDGM",
        domain="latent",
        expected_shape=None,
        datasets=datasets,
        splits_spec={
            "test": "splits/test  vlen-str  held-out patient IDs (shared across folds).",
            "cv": "splits/cv/fold_{0..K-1}/{train,val}  vlen-str  per-fold partition.",
        },
        extras={
            "latent_channels": str(UCSF_PDGM_LATENT_CHANNELS),
            "latent_spatial": str(UCSF_PDGM_LATENT_SPATIAL),
            "image_native_shape": str(UCSF_PDGM_IMAGE_NATIVE_SHAPE),
            "image_padded_shape": str(UCSF_PDGM_IMAGE_PADDED_SHAPE),
            "mask_output_channels": str(mask_output_channels),
            "intensity_policy": "MAISI percentile [0,99.5] -> [0,1] on the encoder input.",
            "decoding_note": "Use DepthPad(after=5) to crop the depth axis back to 155.",
        },
    )


# Default manifest for the canonical v1 run (four sequences, NETC/ED/ET soft mask).
UCSF_PDGM_LATENT_DEFAULT_MODALITIES: list[str] = ["t1pre", "t1c", "t2", "flair"]
UCSF_PDGM_LATENT_MANIFEST: H5Manifest = _build_manifest(
    modalities=UCSF_PDGM_LATENT_DEFAULT_MODALITIES,
    mask_output_channels=3,
)


def build_latent_manifest(
    modalities: list[str],
    mask_output_channels: int,
) -> H5Manifest:
    """Public wrapper around :func:`_build_manifest` for the converter."""
    return _build_manifest(modalities=modalities, mask_output_channels=mask_output_channels)

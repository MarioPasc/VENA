"""Cohort-neutral manifest for the latent-domain H5 cache (schema v2.0.0).

The cache mirrors the image-domain layout (one row per scan, stacked
``chunks=(1, …)`` datasets, ``vlen-str`` IDs and splits) but stores MAISI-V2
VAE-GAN latents instead of raw intensities, plus a MAISI-space tumour mask.

Dimensions are fixed by the common brain-centred crop box
``(192, 224, 192)`` and the autoencoder's 4× spatial compression factor:

* box shape:    ``(192, 224, 192)`` → :data:`LATENT_CROP_BOX`;
* latent shape: ``(48, 56, 48)`` → :data:`LATENT_SPATIAL` with ``C = 4`` → :data:`LATENT_CHANNELS`.

Modalities are listed *dynamically* by the converter: the manifest is built
at write time with the subset declared in the routine config so a v1 run
encoding only ``{t1pre, t1c, t2, flair}`` does not block a later v2 run
that adds ``adc`` and ``swi`` on the same schema version.

Metadata fields are cohort-specific: UCSF-PDGM carries clinical variables;
BraTS-GLI (and other metadata-free cohorts) produce no ``metadata/*``
datasets. Pass ``metadata_fields=[]`` for those cohorts.
"""

from __future__ import annotations

from vena.data.h5.shared import DatasetSpec, H5Manifest

LATENT_SCHEMA_VERSION: str = "2.0.0"

# Common brain-centred crop box applied to every cohort's native volumes.
# The MAISI VAE compresses 4× along each axis.
LATENT_SPATIAL: tuple[int, int, int] = (48, 56, 48)
LATENT_CHANNELS: int = 4

# The padded/box shape that all native volumes are cropped to before encoding.
LATENT_CROP_BOX: tuple[int, int, int] = (192, 224, 192)

# Same MAISI integer codes table that lives in
# src/vena/model/autoencoder/maisi/configs/modality_mapping.json. Mirrored
# here so the manifest stays free of cross-package file reads at import.
LATENT_SEQUENCE_MAP: dict[str, str] = {
    "t1pre": "mri_t1",
    "t1c": "mri_t1c",
    "t2": "mri_t2",
    "flair": "mri_flair",
    "adc": "mri_adc",
    "swi": "mri_swi",
}


def _build_manifest(
    cohort: str,
    modalities: list[str],
    mask_output_channels: int,
    metadata_fields: list[dict[str, str]] | None = None,
) -> H5Manifest:
    """Build the manifest for one encoder run.

    Parameters
    ----------
    cohort : str
        Source cohort tag (e.g. ``"UCSF-PDGM"``, ``"BraTS-GLI"``).
    modalities : list[str]
        Subset of :data:`LATENT_SEQUENCE_MAP` keys to allocate
        latent datasets for.
    mask_output_channels : int
        Number of channels produced by the mask downsampler. Determines the
        spatial shape declared for ``masks/tumor_latent``.
    metadata_fields : list[dict[str, str]] | None
        Per-cohort metadata dataset specs. Each entry carries ``path``,
        ``dtype``, ``units``, and ``description``. Pass ``None`` or ``[]``
        for metadata-free cohorts (e.g. BraTS-GLI); those cohorts produce
        no ``metadata/*`` datasets in the latent H5.
    """
    unknown = [m for m in modalities if m not in LATENT_SEQUENCE_MAP]
    if unknown:
        raise ValueError(
            f"unknown modalities {unknown}; available: {sorted(LATENT_SEQUENCE_MAP)}"
        )
    if not modalities:
        raise ValueError("at least one modality must be requested")

    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description=f"Scan/patient identifier for cohort {cohort!r}.",
            leading_dim="n_scans",
        ),
    ]
    for slug in modalities:
        mapped = LATENT_SEQUENCE_MAP[slug]
        datasets.append(
            DatasetSpec(
                path=f"latents/{slug}",
                dtype="float32",
                kind="image",  # spatial array; reuse "image" kind for shape validation
                units="latent_au",
                description=(
                    f"MAISI-V2 VAE-GAN latent of {slug} (MAISI modality slug {mapped!r}); "
                    "encoded from the common brain-centred crop box."
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

    # CSR patient grouping — always present (copied from source image H5).
    datasets.append(
        DatasetSpec(
            path="patients/offsets",
            dtype="int32",
            kind="metadata",
            units="dimensionless",
            description=(
                "CSR offsets, length n_patients+1; scans of patient k are "
                "rows [offsets[k]:offsets[k+1]]."
            ),
            leading_dim=None,
        )
    )
    datasets.append(
        DatasetSpec(
            path="patients/keys",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description="Unique patient keys, length n_patients, in offset order.",
            leading_dim=None,
        )
    )

    # Cohort-specific metadata (may be empty for BraTS-GLI etc.).
    for field in (metadata_fields or []):
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

    # ``expected_shape`` is set to None: H5Manifest shape checks assume
    # (N, *spatial) but latents have an extra channel axis (N, C, h, w, d).
    # The converter asserts the exact (C, h, w, d) shape itself at
    # create_stacked time.
    return H5Manifest(
        schema_version=LATENT_SCHEMA_VERSION,
        cohort=cohort,
        domain="latent",
        expected_shape=None,
        datasets=datasets,
        splits_spec={
            "test": "splits/test  vlen-str  held-out patient IDs (shared across folds).",
            "cv": "splits/cv/fold_{0..K-1}/{train,val}  vlen-str  per-fold partition.",
        },
        extras={
            "latent_channels": str(LATENT_CHANNELS),
            "latent_spatial": str(LATENT_SPATIAL),
            "crop_box": str(LATENT_CROP_BOX),
            "mask_output_channels": str(mask_output_channels),
            "intensity_policy": (
                "MAISI percentile [0, 99.5] -> [0, 1] on the brain-centred box; "
                "foreground_only=True for skull-stripped volumes."
            ),
            "decoding_note": (
                "Decode returns the box volume (192, 224, 192); compare against "
                "the source cropped to the same box via apply_crop_pad."
            ),
        },
    )


def build_latent_manifest(
    modalities: list[str],
    mask_output_channels: int,
    cohort: str = "UCSF-PDGM",
    metadata_fields: list[dict[str, str]] | None = None,
) -> H5Manifest:
    """Public wrapper around :func:`_build_manifest` for the converter.

    Parameters
    ----------
    modalities : list[str]
        Subset of :data:`LATENT_SEQUENCE_MAP` keys.
    mask_output_channels : int
        Number of channels produced by the mask downsampler.
    cohort : str
        Source cohort tag; defaults to ``"UCSF-PDGM"`` for back-compat.
    metadata_fields : list[dict[str, str]] | None
        Per-cohort metadata specs; ``None`` or ``[]`` for metadata-free cohorts.
    """
    return _build_manifest(
        cohort=cohort,
        modalities=modalities,
        mask_output_channels=mask_output_channels,
        metadata_fields=metadata_fields,
    )

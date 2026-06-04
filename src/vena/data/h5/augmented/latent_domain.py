"""Latent-domain H5 schema for the offline augmentation bank.

The bank's latent H5 is produced by piping the aug-image H5 through
:class:`vena.data.h5.latent_domain.LatentH5Converter` with the new
``aug_mode=True`` flag. The schema mirrors the clean-latent layout one for
one with two omissions and three additions:

* **omitted**: ``patients/offsets``, ``patients/keys`` (CSR), and
  ``splits/*``. Partitioning is the data module's job via
  ``source_row_index`` ↔ the clean latent H5's splits.
* **added**: ``source_row_index``, ``variants``, ``aug_params_json`` per-row
  (carried through from the aug-image H5).

The latent spatial constants are the same as the clean cache:
``(C=4, H=48, W=56, D=48)``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py

from vena.data.h5.latent_domain.manifest import (
    LATENT_CHANNELS,
    LATENT_SPATIAL,
)
from vena.data.h5.shared import DatasetSpec, H5Manifest, validate_h5
from vena.data.h5.shared.exceptions import H5ValidationError

logger = logging.getLogger(__name__)

AUG_LATENT_SCHEMA_VERSION: str = "0.1.0"
"""Schema version of the augmented latent H5."""

_AUG_LATENT_REQUIRED_AUG_ROOT_ATTRS: tuple[str, ...] = (
    "source_aug_image_h5_path",
    "source_aug_image_h5_sha256",
    "aug_config_sha256",
    "variants_json",
)


def build_aug_latent_manifest(
    cohort: str,
    modalities: list[str],
    mask_output_channels: int,
) -> H5Manifest:
    """Build the manifest for one cohort's augmented latent H5.

    Parameters
    ----------
    cohort : str
        Source cohort tag (must match the aug-image H5's ``cohort`` root attr).
    modalities : list[str]
        Modalities written into ``latents/<m>``.
    mask_output_channels : int
        Number of channels produced by the mask downsampler (3 for the
        default ``per_class_avg_pool`` covering NETC/ED/ET).

    Returns
    -------
    H5Manifest
        Frozen manifest with ``domain="latent"`` and
        ``expected_shape=None`` (latent and mask share a leading channel axis
        that differs across datasets; spatial shape is validated per-dataset
        by the producer instead).
    """
    if not modalities:
        raise ValueError("at least one modality must be requested")

    datasets: list[DatasetSpec] = [
        DatasetSpec(
            path="ids",
            dtype="vlen-str",
            kind="id",
            units="dimensionless",
            description=(
                f"Source scan identifier for cohort {cohort!r}; matches the "
                "aug-image H5 row-by-row so `LatentH5Dataset._idx_by_id` "
                "works unchanged."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="source_row_index",
            dtype="int32",
            kind="metadata",
            units="dimensionless",
            description=(
                "Row index into the clean image H5's `ids` array; carried "
                "through from the aug-image H5."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="variants",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description="Per-row augmentation variant tag (`v1`..`v4`).",
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="aug_params_json",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description=(
                "JSON-encoded sampled hyperparameters of every transform that fired for this row."
            ),
            leading_dim="n_scans",
        ),
    ]
    for slug in modalities:
        datasets.append(
            DatasetSpec(
                path=f"latents/{slug}",
                dtype="float32",
                kind="image",
                units="latent_au",
                description=(
                    f"MAISI-V2 VAE-GAN latent of augmented {slug} "
                    f"(channels={LATENT_CHANNELS}, spatial={LATENT_SPATIAL})."
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
                "Soft tumour-label map in MAISI latent space; per-class "
                f"avg-pool with {mask_output_channels} channels (NETC, ED, ET)."
            ),
            leading_dim="n_scans",
        )
    )

    return H5Manifest(
        schema_version=AUG_LATENT_SCHEMA_VERSION,
        cohort=cohort,
        domain="latent",
        expected_shape=None,
        datasets=datasets,
        extras={
            "augmentation_role": "latent_aug",
            "csr_omitted": "patients/offsets,patients/keys",
            "splits_omitted": "splits/*",
        },
    )


def validate_aug_latent_h5(
    path: Path | str,
    cohort: str,
    modalities: list[str],
    mask_output_channels: int,
) -> list[str]:
    """Return human-readable violations for an aug-latent H5; empty = valid."""
    manifest = build_aug_latent_manifest(cohort, modalities, mask_output_channels)
    violations = validate_h5(path, manifest)
    if violations and violations[0].startswith("file does not exist"):
        return violations

    with h5py.File(path, "r") as f:
        for attr in _AUG_LATENT_REQUIRED_AUG_ROOT_ATTRS:
            if attr not in f.attrs:
                violations.append(f"missing aug-specific root attr: {attr}")
        for slug in modalities:
            path_in = f"latents/{slug}"
            if path_in not in f:
                continue
            dset = f[path_in]
            expected_per_row = (LATENT_CHANNELS, *LATENT_SPATIAL)
            if tuple(dset.shape[1:]) != expected_per_row:
                violations.append(
                    f"{path_in}: per-row shape {dset.shape[1:]} != {expected_per_row}"
                )
        if "masks/tumor_latent" in f:
            mdset = f["masks/tumor_latent"]
            expected_mask_per_row = (mask_output_channels, *LATENT_SPATIAL)
            if tuple(mdset.shape[1:]) != expected_mask_per_row:
                violations.append(
                    f"masks/tumor_latent: per-row shape {mdset.shape[1:]} "
                    f"!= {expected_mask_per_row}"
                )
        # Forbidden groups for the aug-latent H5 (splits / CSR live on the
        # clean latent H5 only).
        for forbidden in ("patients", "splits"):
            if forbidden in f:
                violations.append(
                    f"aug-latent H5 must not contain `{forbidden}/*` "
                    "(partitioning lives on the clean latent H5)"
                )
        if "variants_json" in f.attrs:
            try:
                _ = json.loads(str(f.attrs["variants_json"]))
            except json.JSONDecodeError as exc:
                violations.append(f"variants_json failed to parse: {exc}")
    return violations


def assert_aug_latent_h5_valid(
    path: Path | str,
    cohort: str,
    modalities: list[str],
    mask_output_channels: int,
) -> None:
    """Raise :class:`H5ValidationError` listing every violation; succeed silently."""
    violations = validate_aug_latent_h5(path, cohort, modalities, mask_output_channels)
    if violations:
        joined = "\n  - ".join(violations)
        raise H5ValidationError(
            f"Aug-latent H5 failed validation for cohort {cohort!r} "
            f"(schema v{AUG_LATENT_SCHEMA_VERSION}):\n  - {joined}"
        )
    logger.debug("aug-latent H5 valid: %s", path)

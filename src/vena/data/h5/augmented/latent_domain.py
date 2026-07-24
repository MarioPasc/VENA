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
import numpy as np

from vena.data.h5.latent_domain.manifest import (
    LATENT_CHANNELS,
    LATENT_SPATIAL,
)
from vena.data.h5.shared import DatasetSpec, H5Manifest, validate_h5
from vena.data.h5.shared.exceptions import H5ValidationError


def _numpy_int8(dset: h5py.Dataset) -> bool:
    return dset.dtype == np.dtype("int8")


logger = logging.getLogger(__name__)

AUG_LATENT_SCHEMA_VERSION: str = "0.2.0"
"""Schema version of the augmented latent H5.

0.2.0 — 2026-06-19: added the conditional ``masks/brain_latent`` check
(required when root attr ``produced_by_brain_to_latent == True``). Pre-0.2.0
files are accepted as long as their ``masks/brain_latent``, if present,
satisfies the (N, 1, *LATENT_SPATIAL) int8 layout.
"""

AUG_LATENT_SCHEMA_VERSION_SOFT: str = "0.3.0"
"""Schema version stamped on the aug-latent H5 after writing ``masks/tumor_latent_soft``.

This is additive: aug-latent H5 files that have NOT been processed by the
``mask_derive_aug`` routine retain ``schema_version = "0.2.0"`` and continue
to validate against :func:`validate_aug_latent_h5`.  Only once the group
``masks/tumor_latent_soft`` is written does the root ``schema_version``
advance to ``"0.3.0"``.

The write is performed by :class:`routines.segmentation.mask_derive_aug.engine.MaskDeriveAugEngine`;
validated by :func:`assert_aug_latent_soft_mask_group_valid`.
"""

# Expected per-row shape of the soft mask group: (2 channels, *spatial).
_AUG_SOFT_MASK_ROW_SHAPE: tuple[int, ...] = (2, *LATENT_SPATIAL)

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
    # `masks/brain_latent` is written by routines/encode/brain_to_latent as a
    # separate post-pass, so it is NOT in the manifest's datasets list (its
    # absence on a pre-post-pass H5 would otherwise trip the shared validator).
    # `validate_aug_latent_h5` enforces its presence + shape conditionally on
    # the root attr `produced_by_brain_to_latent == True`.

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
        # Conditional check: when the producer recorded the brain-to-latent
        # post-pass having run, `masks/brain_latent` must be present with the
        # canonical (N, 1, *LATENT_SPATIAL) int8 layout.
        if bool(f.attrs.get("produced_by_brain_to_latent", False)):
            if "masks/brain_latent" not in f:
                violations.append(
                    "produced_by_brain_to_latent=True but `masks/brain_latent` is missing"
                )
            else:
                bdset = f["masks/brain_latent"]
                expected_brain_per_row = (1, *LATENT_SPATIAL)
                if tuple(bdset.shape[1:]) != expected_brain_per_row:
                    violations.append(
                        f"masks/brain_latent: per-row shape {bdset.shape[1:]} "
                        f"!= {expected_brain_per_row}"
                    )
                if not _numpy_int8(bdset):
                    violations.append(f"masks/brain_latent: dtype {bdset.dtype} != int8")
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


# ---------------------------------------------------------------------------
# Soft-mask group validator (additive — called after mask_derive_aug write)
# ---------------------------------------------------------------------------

_REQUIRED_DATASET_ATTRS: tuple[str, ...] = ("units", "description", "dtype", "leading_dim")


def validate_aug_latent_soft_mask_group(
    path: Path | str,
    *,
    group: str = "masks/tumor_latent_soft",
) -> list[str]:
    """Validate the additive soft-mask group in an aug-latent H5.

    This validator is separate from :func:`validate_aug_latent_h5` (which uses
    the manifest) so that un-processed ``0.2.0`` aug-latent H5s continue to
    pass their own manifest check unchanged.  Call this function only *after*
    the group has been written by :class:`MaskDeriveAugEngine`.

    Parameters
    ----------
    path : Path or str
        Path to an aug-latent H5 file.
    group : str
        Dataset path to validate; defaults to ``"masks/tumor_latent_soft"``.

    Returns
    -------
    list[str]
        Empty list when the group is valid; non-empty on violations.
    """
    path = Path(path)
    violations: list[str] = []

    if not path.exists():
        return [f"file does not exist: {path}"]

    with h5py.File(path, "r") as f:
        # Schema version must be one of the two supported values.
        sv = str(f.attrs.get("schema_version", ""))
        supported = {AUG_LATENT_SCHEMA_VERSION, AUG_LATENT_SCHEMA_VERSION_SOFT}
        if sv not in supported:
            violations.append(
                f"schema_version {sv!r} not in "
                f"{{{AUG_LATENT_SCHEMA_VERSION!r}, {AUG_LATENT_SCHEMA_VERSION_SOFT!r}}}"
            )

        # mask_source root attr must be present.
        if "mask_source" not in f.attrs:
            violations.append("missing root attr: mask_source")

        # The group itself must exist.
        if group not in f:
            violations.append(f"missing group/dataset: {group!r}")
            return violations  # no further checks are possible

        dset = f[group]
        if not isinstance(dset, h5py.Dataset):
            violations.append(f"{group}: expected Dataset, got {type(dset).__name__}")
            return violations

        # Shape: (N, 2, *LATENT_SPATIAL) — 5 dimensions total.
        if dset.ndim != 5 or tuple(dset.shape[1:]) != _AUG_SOFT_MASK_ROW_SHAPE:
            violations.append(
                f"{group}: per-row shape {tuple(dset.shape[1:])} != {_AUG_SOFT_MASK_ROW_SHAPE}"
            )

        # Dtype must be float32.
        if dset.dtype != np.dtype("float32"):
            violations.append(f"{group}: dtype {dset.dtype} != float32")

        # Required self-describing attrs (principle 4).
        for attr in _REQUIRED_DATASET_ATTRS:
            if attr not in dset.attrs:
                violations.append(f"{group}: missing dataset attr: {attr}")

        # Nesting and range check on first row (cheap sanity).
        if dset.ndim == 4 and dset.shape[0] > 0 and dset.dtype == np.dtype("float32"):
            first_row = dset[0]  # (2, H, W, D)
            tc_ch = first_row[0]
            netc_ch = first_row[1]
            # TC ≥ NETC elementwise (NETC is a subset of TC).
            max_violation = float(np.maximum(netc_ch - tc_ch, 0.0).max())
            if max_violation > 1e-4:
                violations.append(
                    f"{group}: nesting violated on row 0 — "
                    f"max(NETC − TC) = {max_violation:.6f} > 1e-4"
                )
            # Values must be in [0, 1].
            if float(first_row.min()) < -1e-4 or float(first_row.max()) > 1.0 + 1e-4:
                violations.append(
                    f"{group}: row 0 values outside [0, 1] "
                    f"(min={first_row.min():.4f}, max={first_row.max():.4f})"
                )

    return violations


def assert_aug_latent_soft_mask_group_valid(
    path: Path | str,
    *,
    group: str = "masks/tumor_latent_soft",
) -> None:
    """Raise :class:`~vena.data.h5.shared.exceptions.H5ValidationError` on violations; succeed silently."""
    from vena.data.h5.shared.exceptions import H5ValidationError as _H5Err

    violations = validate_aug_latent_soft_mask_group(path, group=group)
    if violations:
        joined = "\n  - ".join(violations)
        raise _H5Err(
            f"Aug-latent soft-mask group {group!r} failed validation in {path}:\n  - {joined}"
        )
    logger.debug("aug-latent soft-mask group valid: %s / %s", path, group)

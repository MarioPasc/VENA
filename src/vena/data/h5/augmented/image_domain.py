"""Image-domain H5 schema for the offline augmentation bank.

The bank stores one row per (source scan × variant). For
``role: cv`` cohorts the source scan IDs are intersected with both the
cohort_dedup allowlist and ``splits/test`` (test patients are excluded);
``role: test_only`` cohorts are not augmented.

Schema invariants enforced by the producer-side validator
(``assert_aug_image_h5_valid``):

* ``schema_version = AUG_IMAGE_SCHEMA_VERSION``, ``cohort = <cohort>``,
  ``domain = "image"`` (the literal allowed by
  :class:`vena.data.h5.shared.H5Manifest`).
* Spatial shape is fixed at :data:`AUG_IMAGE_CROP_BOX` for **every**
  cohort, so the downstream encode is a no-op on the crop axis. This is
  unlike the clean image H5, which stores at native cohort shape.
* The schema-v2 root attrs required by
  :meth:`vena.data.h5.latent_domain.convert.LatentH5Converter._assert_source_compatibility`
  are all present and copied from the source H5:
  ``crop_box``, ``orientation``, ``split_role``, ``longitudinal``,
  ``label_system``. ``crop/origin`` is a per-row dataset and is all-zeros
  for every variant (the H5 itself IS the crop box).
* Aug-specific provenance attrs are also present:
  ``source_image_h5_path``, ``source_image_h5_sha256``,
  ``aug_config_json``, ``aug_config_sha256``, ``variants_json``,
  ``seed``, ``world_size``, ``rank``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py

from vena.data.h5.shared import DatasetSpec, H5Manifest, validate_h5
from vena.data.h5.shared.exceptions import H5ValidationError

logger = logging.getLogger(__name__)

AUG_IMAGE_SCHEMA_VERSION: str = "0.1.0"
"""Schema version of the augmented image-domain H5."""

AUG_IMAGE_CROP_BOX: tuple[int, int, int] = (192, 224, 192)
"""Common brain-centred crop box, identical to the encode pipeline's box.

The bank-builder writes every aug row already cropped to this shape, so the
latent converter's crop+pad step is an identity operation and the v4 elastic
+ affine warp does not have to track a moving ``crop/origin``.
"""

_AUG_IMAGE_REQUIRED_AUG_ROOT_ATTRS: tuple[str, ...] = (
    "source_image_h5_path",
    "source_image_h5_sha256",
    "aug_config_json",
    "aug_config_sha256",
    "variants_json",
    "seed",
    "world_size",
    "rank",
)


def build_aug_image_manifest(
    cohort: str,
    modalities: list[str],
) -> H5Manifest:
    """Build the manifest for one cohort's augmented image-domain H5.

    Parameters
    ----------
    cohort : str
        Source cohort tag (must match the source H5's ``cohort`` root attr).
    modalities : list[str]
        Modalities written into ``images/<m>``. Every input-only variant
        copies ``t1c`` from the source row verbatim; v4 transforms it
        jointly with the other modalities — in both cases all listed
        modalities are stored.

    Returns
    -------
    H5Manifest
        Frozen manifest with ``domain="image"`` and
        ``expected_shape=AUG_IMAGE_CROP_BOX``.
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
                f"Source scan identifier for cohort {cohort!r}. Multiple rows "
                "may share the same scan ID (one per variant)."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="source_row_index",
            dtype="int32",
            kind="metadata",
            units="dimensionless",
            description=(
                "Row index into the clean image H5's `ids` array; lets the "
                "data module join augmented rows back to the splits stored on "
                "the clean latent H5."
            ),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="variants",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description=("Per-row augmentation variant tag (e.g. `v1`, `v2`, `v3`, `v4`)."),
            leading_dim="n_scans",
        ),
        DatasetSpec(
            path="aug_params_json",
            dtype="vlen-str",
            kind="metadata",
            units="dimensionless",
            description=(
                "JSON-encoded sampled hyperparameters of every transform that "
                "fired for this row (random seed-decoded values, post-sampling)."
            ),
            leading_dim="n_scans",
        ),
    ]
    for slug in modalities:
        datasets.append(
            DatasetSpec(
                path=f"images/{slug}",
                dtype="float32",
                kind="image",
                units="intensity_au",
                description=(
                    f"Augmented {slug} volume, cropped to the common brain box "
                    f"{AUG_IMAGE_CROP_BOX}."
                ),
                leading_dim="n_scans",
            )
        )
    datasets.append(
        DatasetSpec(
            path="masks/tumor",
            dtype="int8",
            kind="mask",
            units="label",
            description=(
                "BraTS tumour label map, warped jointly with images for v4; "
                "copy of the source row for v1/v2/v3."
            ),
            leading_dim="n_scans",
        )
    )

    return H5Manifest(
        schema_version=AUG_IMAGE_SCHEMA_VERSION,
        cohort=cohort,
        domain="image",
        expected_shape=AUG_IMAGE_CROP_BOX,
        datasets=datasets,
        extras={
            "augmentation_role": "image_aug",
            "crop_strategy": (
                "applied once in the bank-builder; latent converter sees "
                "crop_origin=(0,0,0) and crop_box=native shape"
            ),
        },
    )


def validate_aug_image_h5(path: Path | str, cohort: str, modalities: list[str]) -> list[str]:
    """Return human-readable violations for an aug-image H5; empty list = valid.

    Combines the shared structural check with the aug-specific provenance
    root-attr requirements and the ``crop/origin == 0`` invariant.
    """
    manifest = build_aug_image_manifest(cohort, modalities)
    violations = validate_h5(path, manifest)
    if violations and violations[0].startswith("file does not exist"):
        return violations

    with h5py.File(path, "r") as f:
        for attr in _AUG_IMAGE_REQUIRED_AUG_ROOT_ATTRS:
            if attr not in f.attrs:
                violations.append(f"missing aug-specific root attr: {attr}")
        if "crop/origin" not in f:
            violations.append("missing crop/origin dataset")
        else:
            origins = f["crop/origin"][...]
            if origins.shape[1:] != (3,):
                violations.append(f"crop/origin: expected shape (N, 3); got {origins.shape}")
            elif (origins != 0).any():
                violations.append(
                    "crop/origin: every row must be (0, 0, 0) for an aug-image H5 "
                    f"(found non-zero rows: {int((origins != 0).any(axis=1).sum())})"
                )
        if "variants_json" in f.attrs:
            try:
                _ = json.loads(str(f.attrs["variants_json"]))
            except json.JSONDecodeError as exc:
                violations.append(f"variants_json failed to parse: {exc}")
    return violations


def assert_aug_image_h5_valid(
    path: Path | str,
    cohort: str,
    modalities: list[str],
) -> None:
    """Raise :class:`H5ValidationError` listing every violation; succeed silently."""
    violations = validate_aug_image_h5(path, cohort, modalities)
    if violations:
        joined = "\n  - ".join(violations)
        raise H5ValidationError(
            f"Aug-image H5 failed validation for cohort {cohort!r} "
            f"(schema v{AUG_IMAGE_SCHEMA_VERSION}):\n  - {joined}"
        )
    logger.debug("aug-image H5 valid: %s", path)

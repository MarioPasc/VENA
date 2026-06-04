"""H5 schemas for the offline-augmentation bank (image + latent domain).

Two parallel artefacts per CV cohort, both produced by
:mod:`routines.offline_aug.maisi`:

* ``<COHORT>_image_aug.h5`` — per-row image-domain volumes of every
  augmented variant (`v1`..`v4`), the warped tumour mask (v4 only; copy
  for v1/v2/v3), and the per-row sampled hyperparameters of every
  transform that fired. Shape is unified across cohorts at the common
  brain-centred crop box ``(192, 224, 192)``; the bank-builder applies
  the crop once, so :class:`vena.data.h5.latent_domain.LatentH5Converter`
  sees ``crop_origin == (0, 0, 0)`` and its own crop+pad becomes a no-op.
* ``<COHORT>_latents_aug.h5`` — the encoded latents produced by feeding
  the image-aug H5 through ``LatentH5Converter`` in ``aug_mode=True``.
  No CSR (``patients/*``), no ``splits/*`` — partitioning is the data
  module's job via ``source_row_index`` ↔ clean-latent-H5 splits.

Both manifests piggyback on the shared :mod:`vena.data.h5.shared`
validator. See ``.claude/rules/h5-design-principles.md`` for the
self-describing-H5 contract.
"""

from __future__ import annotations

from vena.data.h5.augmented.image_domain import (
    AUG_IMAGE_CROP_BOX,
    AUG_IMAGE_SCHEMA_VERSION,
    assert_aug_image_h5_valid,
    build_aug_image_manifest,
    validate_aug_image_h5,
)
from vena.data.h5.augmented.latent_domain import (
    AUG_LATENT_SCHEMA_VERSION,
    assert_aug_latent_h5_valid,
    build_aug_latent_manifest,
    validate_aug_latent_h5,
)

__all__ = [
    "AUG_IMAGE_CROP_BOX",
    "AUG_IMAGE_SCHEMA_VERSION",
    "AUG_LATENT_SCHEMA_VERSION",
    "assert_aug_image_h5_valid",
    "assert_aug_latent_h5_valid",
    "build_aug_image_manifest",
    "build_aug_latent_manifest",
    "validate_aug_image_h5",
    "validate_aug_latent_h5",
]

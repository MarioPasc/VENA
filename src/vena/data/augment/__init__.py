"""Latent-space augmentation for VENA flow-matching training.

Provides paired image/latent operators (one class per augmentation), a
YAML-driven :class:`AugmentationPipeline` that composes them per sample, and
a :class:`AugmentationTracker` callback that records the per-epoch counts of
each applied augmentation combination.

The operators are shared with
:mod:`vena.preflight.latent_aug_equivariance`, which empirically verifies that
``T_image(D(z)) ≈ D(T_latent(z))`` per augmentation before it is admitted into
the runtime pipeline.

Public entry points:

- :class:`LatentAugmentation` — abstract base.
- :class:`AugmentationPipeline` — composes a list of augmentations.
- :func:`build_pipeline_from_yaml` — YAML loader (+ optional preflight gate).
- :class:`AugmentationTracker` — Lightning callback writing
  ``metrics/augmentations_per_epoch.csv``.
- :data:`REGISTRY` — name → class mapping for the YAML loader.
- :class:`LatentAugmentationError` — module exception.
"""

from __future__ import annotations

from vena.data.augment.base import (
    LatentAugmentation,
    LatentAugmentationError,
)
from vena.data.augment.config import (
    AugmentationConfig,
    AugmentationEntryConfig,
    build_pipeline_from_yaml,
)
from vena.data.augment.pipeline import AugmentationPipeline
from vena.data.augment.tracker import AugmentationTracker
from vena.data.augment.transforms import REGISTRY

__all__ = [
    "REGISTRY",
    "AugmentationConfig",
    "AugmentationEntryConfig",
    "AugmentationPipeline",
    "AugmentationTracker",
    "LatentAugmentation",
    "LatentAugmentationError",
    "build_pipeline_from_yaml",
]

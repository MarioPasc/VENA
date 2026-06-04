"""Online (latent-space) augmentation for VENA flow-matching training.

Operators in this subpackage are applied to *latents* on the DataLoader
worker process at training time. They are gated by the latent-equivariance
preflight at :mod:`vena.preflight.latent_aug_equivariance` — only transforms
whose latent-domain implementation matches the encoder-decode of the
image-domain transform within the preflight's pass criterion are admitted
(currently ``flip_lr`` and ``translate``).

The image-domain counterpart of each operator (``apply_image``) exists for
the equivariance preflight's verification step and is not used at runtime.

For *offline* image-domain augmentation (bias field, gamma, histogram
shift, noise/blur, elastic, etc.), see :mod:`vena.data.augment.offline`.

Public entry points
-------------------
- :class:`LatentAugmentation` — abstract base.
- :class:`AugmentationPipeline` — composes a list of augmentations.
- :func:`build_pipeline_from_yaml` — YAML loader (+ optional preflight gate).
- :class:`AugmentationTracker` — Lightning callback writing
  ``metrics/augmentations_per_epoch.csv``.
- :class:`VariantTracker` — Lightning callback writing
  ``metrics/variants_per_epoch.csv`` (offline-aug bank variant counts).
- :data:`REGISTRY` — name → class mapping for the YAML loader.
- :class:`LatentAugmentationError` — module exception.
"""

from __future__ import annotations

from vena.data.augment.online.base import (
    LatentAugmentation,
    LatentAugmentationError,
)
from vena.data.augment.online.config import (
    AugmentationConfig,
    AugmentationEntryConfig,
    build_pipeline_from_yaml,
)
from vena.data.augment.online.pipeline import AugmentationPipeline
from vena.data.augment.online.tracker import (
    AugmentationTracker,
    VariantTracker,
)
from vena.data.augment.online.transforms import REGISTRY

__all__ = [
    "REGISTRY",
    "AugmentationConfig",
    "AugmentationEntryConfig",
    "AugmentationPipeline",
    "AugmentationTracker",
    "LatentAugmentation",
    "LatentAugmentationError",
    "VariantTracker",
    "build_pipeline_from_yaml",
]

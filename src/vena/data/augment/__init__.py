"""Augmentation surface for VENA — online (latent) and offline (image domain).

This package is the canonical adapter for both tiers:

* :mod:`vena.data.augment.online` — latent-space transforms that are applied
  on the DataLoader worker process during training. Currently
  ``flip_lr`` + ``translate`` (per the latent-equivariance preflight).
* :mod:`vena.data.augment.offline` — image-domain transforms that are
  applied **once** per scan per variant, encoded through the frozen MAISI
  VAE, and stored in a per-cohort ``<COHORT>_{image,latents}_aug.h5``.
  These are the transforms the VAE is not equivariant to (bias field,
  gamma, histogram shift, noise, blur, anisotropy, elastic + affine).

For backward compatibility the online tier's public symbols are re-exported
here, so ``from vena.data.augment import AugmentationPipeline`` continues
to work unchanged.
"""

from __future__ import annotations

from vena.data.augment.online import (
    REGISTRY,
    AugmentationConfig,
    AugmentationEntryConfig,
    AugmentationPipeline,
    AugmentationTracker,
    LatentAugmentation,
    LatentAugmentationError,
    VariantTracker,
    build_pipeline_from_yaml,
)

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

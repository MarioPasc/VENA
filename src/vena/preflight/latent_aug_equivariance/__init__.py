"""Latent-augmentation equivariance preflight.

Empirically tests, on a held-out subset of patients from each cohort, whether
applying a candidate augmentation directly to a MAISI-V2 latent and then
decoding matches applying the same augmentation to the decoded volume. A
transformation that passes (PSNR ≥ pass.psnr_db AND SSIM ≥ pass.ssim) on
median across patients is admitted into the runtime augmentation pipeline at
training time.

Public entry points:

- :class:`LatentAugEquivarianceConfig` — pydantic config + ``from_yaml``.
- :class:`LatentAugEquivarianceEngine` — orchestrator with ``run() -> Path``.
- :exc:`LatentAugEquivarianceError` — module exception.
"""

from __future__ import annotations

from vena.preflight.latent_aug_equivariance.engine import (
    LatentAugEquivarianceConfig,
    LatentAugEquivarianceEngine,
    LatentAugEquivarianceError,
)

__all__ = [
    "LatentAugEquivarianceConfig",
    "LatentAugEquivarianceEngine",
    "LatentAugEquivarianceError",
]

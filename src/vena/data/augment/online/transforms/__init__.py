"""Concrete augmentation operators + name → class registry.

The registry is the single source of truth consumed by both the YAML loader
(``vena.data.augment.config.build_pipeline_from_yaml``) and the equivariance
preflight (``vena.preflight.latent_aug_equivariance``).
"""

from __future__ import annotations

from vena.data.augment.online.base import LatentAugmentation
from vena.data.augment.online.transforms.flip import FlipLR
from vena.data.augment.online.transforms.gamma import Gamma
from vena.data.augment.online.transforms.rotate import RotateRoll, RotateYaw
from vena.data.augment.online.transforms.translate import Translate

REGISTRY: dict[str, type[LatentAugmentation]] = {
    FlipLR.name: FlipLR,
    Translate.name: Translate,
    RotateYaw.name: RotateYaw,
    RotateRoll.name: RotateRoll,
    Gamma.name: Gamma,
}

__all__ = [
    "REGISTRY",
    "FlipLR",
    "Gamma",
    "RotateRoll",
    "RotateYaw",
    "Translate",
]

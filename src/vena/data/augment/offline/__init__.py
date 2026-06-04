"""Offline (image-domain) augmentation for VENA.

The MAISI-V2 VAE-GAN is not equivariant to bias field, gamma, monotonic
intensity remap, noise, blur, anisotropy, small-angle rotation, or elastic
deformation (per the latent-equivariance preflight at
:mod:`vena.preflight.latent_aug_equivariance`). To still get the OOD
robustness those transforms buy, we apply them in the image domain, encode
once through the frozen MAISI VAE, and cache the result in a per-cohort
``<COHORT>_{image,latents}_aug.h5`` (schema in
:mod:`vena.data.h5.augmented`).

Public entry points
-------------------
- :func:`make_variant` — builder for one of ``v1``..``v4``.
- :data:`VARIANT_NAMES` — the four variant keys in canonical order.
- :class:`MonaiHistogramShift` — TorchIO wrapper around MONAI's
  ``RandHistogramShift`` (TorchIO has no native monotonic-remap).
- :class:`OfflineAugBankBuilder` — orchestrates per-row augment + write
  for one cohort.
"""

from __future__ import annotations

from vena.data.augment.offline.torchio_adapters import MonaiHistogramShift
from vena.data.augment.offline.variants import (
    VARIANT_INPUT_ONLY,
    VARIANT_NAMES,
    make_variant,
)

__all__ = [
    "VARIANT_INPUT_ONLY",
    "VARIANT_NAMES",
    "MonaiHistogramShift",
    "make_variant",
]

# Defer-import bank_builder so the simpler `make_variant` API works even
# without the bank-builder's heavier h5py + MAISI dependency tree.
try:  # pragma: no cover — import is just a re-export
    from vena.data.augment.offline.bank_builder import OfflineAugBankBuilder

    __all__.append("OfflineAugBankBuilder")
except Exception:
    OfflineAugBankBuilder = None  # type: ignore[assignment]

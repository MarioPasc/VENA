"""MAISI-V2 VAE-GAN adapter for VENA.

This subpackage *wraps* the frozen MAISI autoencoder; it never modifies the
external source under ``src/external/`` (see ``.claude/rules/external-deps.md``).
The wrapper exposes:

* :class:`AutoencoderHandle` + :func:`load_autoencoder` — instantiation +
  checkpoint loading + provenance metadata.
* :class:`MaisiEncoder` (``encode``-submodule) — image → latent with
  full-volume try / sliding-window fallback.
* :class:`MaisiDecoder` (``decode``-submodule) — latent → image with the
  same fallback, plus depth crop-back to the original shape.

The spatial compression factor is 4 (three stride-2 stages with paired
res-blocks). For a UCSF-PDGM volume of ``(240, 240, 155)``, encoding first
zero-pads the depth axis to ``160`` and produces a latent of shape
``(4, 60, 60, 40)``; decoding inverts both operations.
"""

from __future__ import annotations

from .exceptions import (
    CheckpointLoadError,
    EncodeOOMError,
    MaisiError,
    ShapeContractError,
)
from .loader import AutoencoderHandle, load_autoencoder
from .preprocessing import DepthPad, crop_to_original, pad_depth_to_multiple_of, percentile_normalise

SPATIAL_COMPRESSION: int = 4
LATENT_CHANNELS: int = 4

__all__ = [
    "LATENT_CHANNELS",
    "SPATIAL_COMPRESSION",
    "AutoencoderHandle",
    "CheckpointLoadError",
    "DepthPad",
    "EncodeOOMError",
    "MaisiError",
    "ShapeContractError",
    "crop_to_original",
    "load_autoencoder",
    "pad_depth_to_multiple_of",
    "percentile_normalise",
]

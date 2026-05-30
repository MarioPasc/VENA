"""Cross-cutting primitives reused across data, model, and routine layers.

This package is the canonical adapter surface for the frozen MAISI-V2 VAE-GAN
and shared decode helpers. Per ``.claude/rules/external-deps.md``, no source
file under ``src/external/`` is ever edited; instead, the VENA-side wrappers
live under ``vena.model.autoencoder.maisi`` and are re-exported from here so
that downstream callers (training, evaluation, preflight, exhaustive
validation) depend on a single import path.

Re-exports
----------
* :class:`MaisiEncoder`, :class:`MaisiDecoder` — VAE encode/decode primitives.
* :func:`load_autoencoder` — instantiate the frozen VAE with provenance.
* :class:`AutoencoderHandle` — dataclass returned by ``load_autoencoder``.
* :func:`percentile_normalise` — canonical intensity normalisation matching
  the MAISI training transform (``lower=0`` / ``upper=99.5``).
* :class:`DepthPad`, :func:`pad_depth_to_multiple_of`,
  :func:`crop_to_original` — depth-axis pad/crop helpers (used by the
  in-process training-time decode proxy).
* :class:`CropPadSpec`, :func:`apply_crop_pad`, :func:`invert_crop_pad` —
  full-volume brain-box crop/pad helpers (used by the exhaustive-validation
  full-volume decode path).

Shared decode entry points live in :mod:`vena.common.decode`.
"""

from __future__ import annotations

from vena.model.autoencoder.maisi import (
    LATENT_CHANNELS,
    SPATIAL_COMPRESSION,
    AutoencoderHandle,
    CheckpointLoadError,
    DepthPad,
    EncodeOOMError,
    MaisiError,
    ShapeContractError,
    crop_to_original,
    load_autoencoder,
    pad_depth_to_multiple_of,
    percentile_normalise,
)
from vena.model.autoencoder.maisi.decode import (
    DecodeMode,
    DecodeResult,
    MaisiDecoder,
)
from vena.model.autoencoder.maisi.encode import EncodeResult, MaisiEncoder
from vena.model.autoencoder.maisi.preprocessing import (
    CropPadSpec,
    apply_crop_pad,
    invert_crop_pad,
)

__all__ = [
    "LATENT_CHANNELS",
    "SPATIAL_COMPRESSION",
    "AutoencoderHandle",
    "CheckpointLoadError",
    "CropPadSpec",
    "DecodeMode",
    "DecodeResult",
    "DepthPad",
    "EncodeOOMError",
    "EncodeResult",
    "MaisiDecoder",
    "MaisiEncoder",
    "MaisiError",
    "ShapeContractError",
    "apply_crop_pad",
    "crop_to_original",
    "invert_crop_pad",
    "load_autoencoder",
    "pad_depth_to_multiple_of",
    "percentile_normalise",
]

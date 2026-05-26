"""Exceptions raised by the MAISI VAE-GAN adapter."""

from __future__ import annotations


class MaisiError(Exception):
    """Base for every exception raised by ``vena.model.autoencoder.maisi``."""


class CheckpointLoadError(MaisiError):
    """Raised when the autoencoder checkpoint cannot be located, parsed, or
    loaded into the instantiated :class:`AutoencoderKlMaisi` (state-dict
    mismatch, missing ``unet_state_dict`` key, etc.)."""


class ShapeContractError(MaisiError):
    """A tensor handed to the encoder, decoder, or mask downsampler does not
    satisfy the documented shape contract (rank, channel count, divisibility
    by the spatial compression factor)."""


class EncodeOOMError(MaisiError):
    """Both the full-volume and sliding-window encode paths exhausted GPU
    memory. Raised after the fallback has also failed so the routine can
    record a clear cause rather than burying it under a CUDA traceback."""

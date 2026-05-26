"""Exceptions raised by mask-downsampler models."""

from __future__ import annotations


class MaskDownsamplerError(Exception):
    """Base for every exception raised by mask-downsampler models.

    Kept narrow on purpose: the downsampler subsystem already shares
    :class:`vena.model.autoencoder.maisi.ShapeContractError` for input-shape
    issues; this class is reserved for *semantic* failures (unknown label
    code, invalid output channel count, registry miss).
    """


class UnknownDownsamplerError(MaskDownsamplerError):
    """Raised when a mask downsampler is requested by name and no model
    with that name is registered."""


class LabelCodeError(MaskDownsamplerError):
    """A mask contained a label code not declared by the downsampler.

    Surfaced as a hard error rather than silently dropped, because a stray
    label usually indicates a different label-set convention than the one
    the model targets (e.g. BraTS-2017 ``{1,2,4}`` vs BraTS-2023 ``{1,2,3}``).
    """

"""Shared decode helpers consumed by training validation and exhaustive val.

Two distinct decode paths exist in VENA. Both wrap :class:`MaisiDecoder` but
differ in the geometry contract:

* :func:`decode_box` — full-volume reconstruction onto the *brain box* defined
  by a :class:`CropPadSpec`. Used by the asynchronous exhaustive-validation
  job (``routines/fm/exhaustive_val``) for image-space PSNR/SSIM against the
  real T1c volume.
* :func:`decode_depth_identity` — depth-axis identity un-pad only (no box
  crop/pad). Used by the in-process training-time image-metric proxy in
  :class:`FMLightningModule` when the H5-stored latent is already padded to
  the next multiple of the VAE depth-compression factor.

Both helpers are intentionally tiny: their value is naming + a single import
path. They do not introduce new behaviour relative to direct calls of
``MaisiDecoder.decode``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, overload

import torch

from vena.model.autoencoder.maisi import (
    SPATIAL_COMPRESSION,
    DepthPad,
)

if TYPE_CHECKING:  # avoid eager imports during package init
    from vena.model.autoencoder.maisi.decode import DecodeResult, MaisiDecoder
    from vena.model.autoencoder.maisi.preprocessing import CropPadSpec


@overload
def decode_box(
    decoder: MaisiDecoder,
    latent: torch.Tensor,
    crop_spec: CropPadSpec,
    *,
    clamp_unit_interval: bool = ...,
    return_seconds: Literal[False] = ...,
) -> torch.Tensor: ...


@overload
def decode_box(
    decoder: MaisiDecoder,
    latent: torch.Tensor,
    crop_spec: CropPadSpec,
    *,
    clamp_unit_interval: bool = ...,
    return_seconds: Literal[True],
) -> tuple[torch.Tensor, float]: ...


def decode_box(
    decoder: MaisiDecoder,
    latent: torch.Tensor,
    crop_spec: CropPadSpec,
    *,
    clamp_unit_interval: bool = True,
    return_seconds: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """Decode a latent into the brain-box volume.

    Parameters
    ----------
    decoder : MaisiDecoder
        Frozen VAE decoder.
    latent : torch.Tensor
        Latent of shape ``(1, C, h, w, d)``.
    crop_spec : CropPadSpec
        Brain-box geometry mapping latent → image space.
    clamp_unit_interval : bool, default True
        If True, clamp the decoded volume to ``[0, 1]`` (matches the VAE
        training-target intensity range).
    return_seconds : bool, default False
        If True, return ``(volume, decode_sec)``; the CUDA stream is
        synchronised on both sides of the decode for accurate timing.

    Returns
    -------
    torch.Tensor
        Decoded volume of shape ``(*target_shape,)`` — the singleton batch
        and channel axes are squeezed.
    """
    device = latent.device
    if return_seconds and device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter() if return_seconds else 0.0
    with torch.inference_mode():
        out = decoder.decode(latent, crop_spec=crop_spec)
    if return_seconds and device.type == "cuda":
        torch.cuda.synchronize(device)
    volume = out.image[0, 0].float()
    if clamp_unit_interval:
        volume = volume.clamp(0.0, 1.0)
    if return_seconds:
        return volume, time.perf_counter() - t0
    return volume


def decode_depth_identity(
    decoder: MaisiDecoder,
    latent: torch.Tensor,
) -> DecodeResult:
    """Decode a latent without any depth-axis padding (identity un-pad).

    The H5 latent cache stores depth-padded latents (image-space depth zero-padded
    to a multiple of :data:`SPATIAL_COMPRESSION` before encoding). When the
    consumer compares two latents that were both stored under the same depth
    convention, the decode does not need to undo the pad — we ask for
    ``before=after=0`` and ``original_depth == padded_depth == latent_depth * 4``.

    Parameters
    ----------
    decoder : MaisiDecoder
        Frozen VAE decoder.
    latent : torch.Tensor
        Latent of shape ``(B, C, h, w, d)``.

    Returns
    -------
    DecodeResult
        Raw decoder output; the caller may extract ``.image``.
    """
    latent_d = int(latent.shape[-1])
    depth = latent_d * SPATIAL_COMPRESSION
    pad = DepthPad(before=0, after=0, original_depth=depth, padded_depth=depth)
    return decoder.decode(latent, pad)


__all__ = ["decode_box", "decode_depth_identity"]

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

A third helper, :func:`partial_decode`, runs a *partial* decode through
``decoder.blocks[0..max_block]`` and captures intermediate activations at
the requested block indices via forward hooks. It is the load-bearing
primitive for the LPL preflight (``routines/preflights/decoder_lpl_profile``)
and for the future S3 stage's gated decoder-feature loss (Berrada et al.
2025, *Boosting Latent Diffusion with Perceptual Objectives*, arXiv:2411.04873).
The function takes the **MONAI** ``MaisiDecoder`` (reachable as
``handle.model.decoder``) — *not* VENA's :class:`MaisiDecoder` wrapper. The
caller is responsible for running ``handle.model.post_quant_conv(latent)``
before invoking ``partial_decode``; folding ``post_quant_conv`` here would
break the "decoder.blocks-only" contract the function name advertises and
force every caller to pass the thicker ``handle.model``. The
:func:`vena.model.fm.lpl.hooks.decoder_feature_extractor` context manager
wraps both steps so most library callers do not see this detail.

These helpers do not introduce new behaviour relative to direct calls of
``MaisiDecoder.decode`` / ``MaisiDecoder.forward``; their value is naming
plus a single import path.
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
    from typing import Any

    from torch import nn

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


def partial_decode(
    decoder: nn.Module,
    latent_after_post_quant: torch.Tensor,
    *,
    blocks: set[int] | frozenset[int],
    max_block: int,
    grad_checkpoint: bool = False,
) -> dict[int, torch.Tensor]:
    """Run the decoder forward through ``blocks[0..max_block]``, capture features.

    The MONAI ``MaisiDecoder.forward`` iterates a flat ``nn.ModuleList`` of
    11 blocks (see ``decoder_block_geometry`` for the canonical VENA
    geometry) and emits ``_empty_cuda_cache`` between every block when
    ``save_mem=True``. That can recycle the storage of a captured activation
    before the caller uses it, so the forward hook installed here clones the
    block output. The clone is load-bearing — without it block-2 / block-5
    capture intermittently reads recycled memory under ``save_mem=True``.

    Parameters
    ----------
    decoder : nn.Module
        The MONAI ``MaisiDecoder`` (``handle.model.decoder``). The caller
        must have already applied ``handle.model.post_quant_conv`` to the
        latent — this function operates *strictly* on the post-quant
        decoder block sequence.
    latent_after_post_quant : torch.Tensor
        Latent of shape ``(B, C_post_quant, h, w, d)``, already passed
        through ``post_quant_conv``.
    blocks : set[int]
        Block indices to capture. Must be a subset of ``{0..max_block}``.
    max_block : int
        Last block index (inclusive) to actually run. The forward is
        truncated to ``decoder.blocks[: max_block + 1]``.
    grad_checkpoint : bool, default False
        If True, wrap the block slice in
        :func:`torch.utils.checkpoint.checkpoint_sequential` with
        ``segments=2`` to trade backward time for activation memory. Hooks
        still fire on the segmented forward; the captured tensors are
        ``.detach().clone()`` of the block output (not gradient-bearing —
        they are intended for downstream feature-distance computation
        rather than re-entry into autograd).

    Returns
    -------
    dict[int, torch.Tensor]
        Mapping from ``block_idx`` to captured activation, shape
        ``(B, C_block, h_block, w_block, d_block)``. The dict's iteration
        order follows the ``blocks`` set's hash order — callers that need
        a deterministic order should iterate ``sorted(out)``.

    Raises
    ------
    ValueError
        If ``max_block`` is out of range, or any element of ``blocks`` is
        outside ``[0, max_block]``.
    """
    n = len(decoder.blocks)  # type: ignore[arg-type]
    if max_block < 0 or max_block >= n:
        raise ValueError(f"max_block={max_block} out of range for decoder with {n} blocks")
    blocks_set: frozenset[int] = frozenset(blocks)
    if not blocks_set:
        raise ValueError("blocks set must be non-empty")
    if min(blocks_set) < 0 or max(blocks_set) > max_block:
        raise ValueError(f"blocks {sorted(blocks_set)} not in valid range [0, {max_block}]")

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def _hook(_module, _inputs, output):
            # Clone defeats the MaisiDecoder's per-block _empty_cuda_cache
            # call (it may recycle the underlying storage before the caller
            # consumes the activation).
            captured[idx] = output.detach().clone()

        return _hook

    handles = []
    for idx in blocks_set:
        h = decoder.blocks[idx].register_forward_hook(_make_hook(idx))  # type: ignore[index]
        handles.append(h)
    try:
        x = latent_after_post_quant
        if grad_checkpoint:
            # Slice + segmenting both bound the activation memory and keep
            # the hook contract intact. ``use_reentrant=False`` matches the
            # MONAI MAISI code's own checkpoint call.
            from torch.utils.checkpoint import checkpoint_sequential

            x = checkpoint_sequential(
                decoder.blocks[: max_block + 1],  # type: ignore[index]
                segments=2,
                input=x,
                use_reentrant=False,
            )
        else:
            for i in range(max_block + 1):
                x = decoder.blocks[i](x)  # type: ignore[index]
    finally:
        for h in handles:
            h.remove()

    return captured


def decoder_block_geometry(
    decoder: nn.Module,
) -> list[dict[str, Any]]:
    """Static enumeration of the decoder block stack — no forward pass.

    For each block in ``decoder.blocks``, return a dict with:

    * ``idx`` — position in the ModuleList.
    * ``type`` — class name (e.g. ``"MaisiResBlock"``, ``"MaisiUpsample"``).
    * ``in_channels`` / ``out_channels`` — when discoverable from the
      block's submodule attributes; ``None`` otherwise.

    Used by :class:`vena.model.fm.lpl.config.LplConfig` to validate that
    a requested readout set ``A`` lies inside the decoder's actual block
    range, and by the LPL preflight to build the per-block magnitude
    curve table.
    """
    out: list[dict[str, Any]] = []
    blocks = decoder.blocks  # type: ignore[attr-defined]
    for idx, block in enumerate(blocks):
        info: dict[str, Any] = {"idx": idx, "type": type(block).__name__}
        # Best-effort channel discovery — MAISI MONAI blocks expose
        # ``in_channels`` / ``out_channels`` on convolutions but not all
        # res-blocks. None is the safe fallback for the consumer.
        for name in ("in_channels", "out_channels"):
            val = getattr(block, name, None)
            info[name] = int(val) if isinstance(val, int) else None
        out.append(info)
    return out


__all__ = [
    "decode_box",
    "decode_depth_identity",
    "decoder_block_geometry",
    "partial_decode",
]

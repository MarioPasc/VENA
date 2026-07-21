"""Image-space preprocessing utilities for the MAISI VAE-GAN.

Three small operations live here:

* ``percentile_normalise`` — MAISI MR expects inputs in ``[0, 1]`` produced by
  the same percentile rescale used by MAISI's own ``VAE_Transform``
  (``lower=0`` / ``upper=99.5`` percentiles → ``[b_min, b_max]``). Replicated
  here so the encoder can be driven from raw H5 intensities without the
  full MAISI transform stack.
* ``pad_depth_to_multiple_of`` — UCSF-PDGM volumes are ``(240, 240, 155)``;
  155 is not a multiple of 4 (the VAE's compression factor). We end-pad the
  depth axis with zeros to the next multiple of ``base`` and remember the
  pad so :func:`crop_to_original` can undo it after decode.
* ``crop_to_original`` — the inverse op, given the recorded pad. Symmetric
  to :func:`pad_depth_to_multiple_of`.

All functions operate on ``torch.Tensor`` in shape ``(B, C, H, W, D)``;
``C`` is unused — preserved verbatim — but checked for sanity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .exceptions import ShapeContractError

_DEFAULT_BASE = 8


@dataclass(frozen=True)
class DepthPad:
    """Padding applied along the depth (last) axis.

    Attributes
    ----------
    before : int
        Voxels prepended to ``D``.
    after : int
        Voxels appended to ``D``.
    original_depth : int
        Pre-pad ``D``; used by :func:`crop_to_original`.
    padded_depth : int
        Post-pad ``D``.
    """

    before: int
    after: int
    original_depth: int
    padded_depth: int


#: Intensity percentile the frozen MAISI encoder used for EVERY production latent
#: cache. All production encode configs set ``percentile_upper: 99.95``
#: (``routines/encode/maisi/configs/picasso.yaml``, ``server3.yaml`` and every
#: ``*_server3.yaml``); only the smoke/``default.yaml`` configs use 99.5. Confirmed
#: by the 2026-07-21 ρ_S normalisation audit: a decode of the cached ``z_t1c`` is
#: self-consistent at 99.95 (PSNR 29.5 dB, ρ_S≈0) but mismatched at 99.5 (20.9 dB,
#: ρ_S 0.66, because 99.5 saturates the enhancing-rim/vessel tail). Any decode-vs-real
#: comparison MUST normalise the reference at this percentile so the decoded prediction
#: (99.95-latent space) and the reference share one intensity space. Do NOT revert to
#: 99.5 — that reintroduces the audit's confound. Single source for eval + analysis.
ENCODER_PERCENTILE_UPPER: float = 99.95


def percentile_normalise(
    x: torch.Tensor,
    lower: float = 0.0,
    upper: float = 99.5,
    b_min: float = 0.0,
    b_max: float = 1.0,
    eps: float = 1e-8,
    foreground_only: bool = False,
    foreground_threshold: float = 0.0,
    mask: torch.Tensor | None = None,
    clip: bool = True,
) -> torch.Tensor:
    """Map ``x`` into ``[b_min, b_max]`` using per-volume percentile clipping.

    Mirrors MAISI's ``ScaleIntensityRangePercentiles`` transform. The
    percentiles are computed independently for each ``(B, C)`` slice so
    multi-modality batches do not contaminate each other.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``(B, C, H, W, D)``; any floating dtype.
    lower, upper : float
        Percentile bounds in ``[0, 100]``.
    b_min, b_max : float
        Output range.
    eps : float
        Numerical guard for empty / constant volumes.
    foreground_only : bool
        When ``True``, compute the percentiles over voxels with intensity
        strictly above ``foreground_threshold`` (typically the skull-strip
        background mask). The clip/scale is still applied to the entire
        volume so background voxels remain zero. Recommended for
        skull-stripped MR (Isensee *et al.* 2021, Reinhold *et al.* 2019).
        Ignored when ``mask`` is provided.
    foreground_threshold : float
        Inclusive lower bound for the foreground mask. Defaults to ``0``
        (anything strictly above 0 is foreground). Ignored when ``mask`` is
        provided.
    mask : torch.Tensor | None
        Optional explicit brain-mask of shape ``(B, 1, H, W, D)`` or
        broadcastable to ``x``. When provided, the foreground voxels are
        taken from ``mask > 0`` and the ``foreground_only`` / ``foreground_threshold``
        heuristic is bypassed. Required for cohorts that store
        z-score-normalised intensities (e.g. BraTS-Africa) where the
        ``x > 0`` heuristic silently excludes the negative half of the
        intra-brain distribution.
    clip : bool
        When ``True`` (default — backwards-compatible) the rescaled volume is
        clamped to ``[0, 1]`` before mapping to ``[b_min, b_max]``. When
        ``False`` the clamp is skipped and super-percentile values keep their
        magnitude (>1 after the affine), preserving the bright tail. Used by
        the v3 normalisation audit (see
        ``.claude/notes/changes/2026-06-22_s1_v3_normalization_exploration.md``)
        to test whether the hard clip is what crushes the T1c gadolinium-
        enhancement signal.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``x``. When ``clip=True`` (default), values
        lie in ``[b_min, b_max]``; when ``clip=False``, the rescale is
        applied without clipping (output may exceed ``b_max`` for voxels
        above the ``upper`` percentile).

    Raises
    ------
    ShapeContractError
        If ``x`` is not a 5-D tensor, or if ``mask`` is provided with an
        incompatible shape.
    """
    if x.ndim != 5:
        raise ShapeContractError(
            f"percentile_normalise expects (B,C,H,W,D); got shape {tuple(x.shape)}"
        )

    if mask is not None:
        if mask.ndim != 5:
            raise ShapeContractError(
                f"percentile_normalise: mask expects (B,1,H,W,D); got shape {tuple(mask.shape)}"
            )
        if mask.shape[0] != x.shape[0] or tuple(mask.shape[2:]) != tuple(x.shape[2:]):
            raise ShapeContractError(
                f"percentile_normalise: mask spatial / batch shape {tuple(mask.shape)} "
                f"incompatible with input {tuple(x.shape)}"
            )
        B, C = x.shape[0], x.shape[1]
        lo = torch.empty((B, C, 1, 1, 1), dtype=x.dtype, device=x.device)
        hi = torch.empty_like(lo)
        # mask: (B, 1, H, W, D) broadcast across channel.
        m_bool = mask > 0
        for b in range(B):
            mb = m_bool[b, 0]
            for c in range(C):
                fg = x[b, c][mb]
                if fg.numel() == 0:
                    lo[b, c, 0, 0, 0] = 0.0
                    hi[b, c, 0, 0, 0] = 1.0
                    continue
                q = torch.tensor([lower / 100.0, upper / 100.0], dtype=fg.dtype, device=fg.device)
                lh = torch.quantile(fg, q)
                lo[b, c, 0, 0, 0] = lh[0]
                hi[b, c, 0, 0, 0] = lh[1]
    elif foreground_only:
        # Compute percentiles per (B, C) slice over the foreground voxels
        # only. We have to loop because torch.quantile does not support a
        # per-row masked variant; the loop is over at most B*C items.
        B, C = x.shape[0], x.shape[1]
        lo = torch.empty((B, C, 1, 1, 1), dtype=x.dtype, device=x.device)
        hi = torch.empty_like(lo)
        for b in range(B):
            for c in range(C):
                vol = x[b, c]
                fg = vol[vol > foreground_threshold]
                if fg.numel() == 0:
                    lo[b, c, 0, 0, 0] = 0.0
                    hi[b, c, 0, 0, 0] = 1.0
                    continue
                q = torch.tensor([lower / 100.0, upper / 100.0], dtype=fg.dtype, device=fg.device)
                lh = torch.quantile(fg, q)
                lo[b, c, 0, 0, 0] = lh[0]
                hi[b, c, 0, 0, 0] = lh[1]
    else:
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        q = torch.tensor([lower / 100.0, upper / 100.0], dtype=flat.dtype, device=flat.device)
        lo_hi = torch.quantile(flat, q, dim=-1)  # (2, B, C)
        lo = lo_hi[0].view(x.shape[0], x.shape[1], 1, 1, 1)
        hi = lo_hi[1].view(x.shape[0], x.shape[1], 1, 1, 1)

    denom = (hi - lo).clamp_min(eps)
    y = (x - lo) / denom
    if clip:
        y = y.clamp(0.0, 1.0)
    return y * (b_max - b_min) + b_min


def pad_depth_to_multiple_of(
    x: torch.Tensor,
    base: int = _DEFAULT_BASE,
) -> tuple[torch.Tensor, DepthPad]:
    """End-pad the depth axis so it is divisible by ``base``.

    MAISI's encoder downsamples 4× along every spatial axis; padding to a
    multiple of 8 leaves a margin so future encoder revisions (one extra
    stride-2 stage) still work. Pad happens *after* the original depth so
    every original voxel keeps the same axial / sagittal / coronal index.
    """
    if x.ndim != 5:
        raise ShapeContractError(
            f"pad_depth_to_multiple_of expects (B,C,H,W,D); got shape {tuple(x.shape)}"
        )
    if base <= 0:
        raise ValueError(f"base must be positive; got {base}")
    d = x.shape[-1]
    remainder = d % base
    after = 0 if remainder == 0 else (base - remainder)
    if after == 0:
        return x, DepthPad(before=0, after=0, original_depth=d, padded_depth=d)
    # F.pad takes (W_left, W_right, H_left, H_right, D_left, D_right) for 3D
    # tensors arranged (..., D, H, W). For (B,C,H,W,D) the last dim is D, so
    # the first two entries pad D. We end-pad: (0, after).
    y = F.pad(x, (0, after, 0, 0, 0, 0), mode="constant", value=0.0)
    return y, DepthPad(before=0, after=after, original_depth=d, padded_depth=d + after)


def crop_to_original(x: torch.Tensor, pad: DepthPad) -> torch.Tensor:
    """Reverse :func:`pad_depth_to_multiple_of` given the recorded ``pad``."""
    if x.ndim != 5:
        raise ShapeContractError(
            f"crop_to_original expects (B,C,H,W,D); got shape {tuple(x.shape)}"
        )
    if x.shape[-1] != pad.padded_depth:
        raise ShapeContractError(
            f"crop_to_original: input depth {x.shape[-1]} != padded_depth {pad.padded_depth}"
        )
    return x[..., pad.before : pad.before + pad.original_depth]


@dataclass(frozen=True)
class CropPadSpec:
    """Brain-centred crop+pad mapping a native RAS volume onto a fixed box.

    The box is defined in canonical-LPS voxel space (axes L→R, P→A, S→I,
    matching the ``(H, W, D)`` order of the stored arrays). ``crop_origin`` is
    the index in the *native* grid at which the box starts; it may be negative
    (the box extends before the native array → zero-pad before) and
    ``crop_origin + target_shape`` may exceed ``native_shape`` (→ zero-pad
    after). On axes where ``native_shape >= target_shape`` the box crops the
    native volume; where ``native_shape < target_shape`` it pads.

    Attributes
    ----------
    crop_origin : tuple[int, int, int]
        Native-grid start index of the box per axis ``(H, W, D)``.
    native_shape : tuple[int, int, int]
        Spatial shape of the source volume.
    target_shape : tuple[int, int, int]
        Common box shape (e.g. ``(192, 224, 192)``).
    """

    crop_origin: tuple[int, int, int]
    native_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]


def apply_crop_pad(x: torch.Tensor, spec: CropPadSpec) -> torch.Tensor:
    """Crop/zero-pad a native-RAS volume onto ``spec.target_shape``.

    Parameters
    ----------
    x : torch.Tensor
        Volume of shape ``(B, C, *spec.native_shape)``.
    spec : CropPadSpec
        Crop/pad geometry.

    Returns
    -------
    torch.Tensor
        Volume of shape ``(B, C, *spec.target_shape)``.

    Raises
    ------
    ShapeContractError
        If ``x`` is not 5-D or its spatial shape disagrees with
        ``spec.native_shape``.
    """
    if x.ndim != 5:
        raise ShapeContractError(f"apply_crop_pad expects (B,C,H,W,D); got shape {tuple(x.shape)}")
    if tuple(x.shape[2:]) != tuple(spec.native_shape):
        raise ShapeContractError(
            f"apply_crop_pad: input spatial {tuple(x.shape[2:])} != "
            f"native_shape {spec.native_shape}"
        )
    o, n, t = spec.crop_origin, spec.native_shape, spec.target_shape
    src: list[tuple[int, int]] = []
    pad: list[tuple[int, int]] = []
    for i in range(3):
        s_start = max(0, o[i])
        s_end = min(n[i], o[i] + t[i])
        before = s_start - o[i]
        copied = max(0, s_end - s_start)
        after = t[i] - before - copied
        src.append((s_start, s_end))
        pad.append((before, after))
    cropped = x[
        :,
        :,
        src[0][0] : src[0][1],
        src[1][0] : src[1][1],
        src[2][0] : src[2][1],
    ]
    # F.pad pads the last spatial axis first: (D_before, D_after, W_*, H_*).
    pad_arg = (
        pad[2][0],
        pad[2][1],
        pad[1][0],
        pad[1][1],
        pad[0][0],
        pad[0][1],
    )
    return F.pad(cropped, pad_arg, mode="constant", value=0.0)


def invert_crop_pad(x: torch.Tensor, spec: CropPadSpec) -> torch.Tensor:
    """Inverse of :func:`apply_crop_pad`: map a box volume back to native shape.

    Padded margins are dropped and native regions the box never covered are
    zero-filled. Used to write box-space predictions back into the native grid;
    metric computation instead crops the *real* image with :func:`apply_crop_pad`.

    Parameters
    ----------
    x : torch.Tensor
        Volume of shape ``(B, C, *spec.target_shape)``.
    spec : CropPadSpec
        Crop/pad geometry.

    Returns
    -------
    torch.Tensor
        Volume of shape ``(B, C, *spec.native_shape)``.
    """
    if x.ndim != 5:
        raise ShapeContractError(f"invert_crop_pad expects (B,C,H,W,D); got shape {tuple(x.shape)}")
    if tuple(x.shape[2:]) != tuple(spec.target_shape):
        raise ShapeContractError(
            f"invert_crop_pad: input spatial {tuple(x.shape[2:])} != "
            f"target_shape {spec.target_shape}"
        )
    o, n, t = spec.crop_origin, spec.native_shape, spec.target_shape
    out = x.new_zeros((x.shape[0], x.shape[1], *n))
    dst: list[tuple[int, int]] = []
    box: list[tuple[int, int]] = []
    for i in range(3):
        d_start = max(0, o[i])
        d_end = min(n[i], o[i] + t[i])
        dst.append((d_start, d_end))
        box.append((d_start - o[i], d_end - o[i]))
    out[
        :,
        :,
        dst[0][0] : dst[0][1],
        dst[1][0] : dst[1][1],
        dst[2][0] : dst[2][1],
    ] = x[
        :,
        :,
        box[0][0] : box[0][1],
        box[1][0] : box[1][1],
        box[2][0] : box[2][1],
    ]
    return out

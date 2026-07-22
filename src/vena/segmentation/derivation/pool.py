"""Avg-pool of image-space soft masks to the MAISI latent grid.

The registration pipeline replicates *exactly* the convention used by
``masks/tumor_latent`` in the latent H5 (see
:mod:`vena.data.h5.latent_domain.convert._encode_loop`):

1. **[Optional]** Crop the native-space volume to the brain-centred box
   ``LATENT_CROP_BOX = (192, 224, 192)`` via :func:`vena.common.apply_crop_pad`.
   ``apply_crop_pad`` requires a 5-D ``(B, C, H, W, D)`` tensor; the channel
   dim is temporarily expanded and then squeezed back.
2. Depth-pad D to a multiple of ``avg_pool_stride * 2`` (= 8 at default
   stride 4), matching :class:`PerClassAvgPoolDownsampler.depth_pad_base`.
   For the canonical crop box D=192 is already divisible by 8 and this step
   is a no-op.
3. :func:`torch.nn.functional.avg_pool3d` with ``kernel_size = stride =
   avg_pool_stride`` on the 4-D ``(C, H, W, D)`` tensor (PyTorch allows the
   unbatched form) → ``(2, 48, 56, 48)``.

**Order is fixed**: sigmoid / temperature MUST be applied before calling
this function.  Pooling raw signed logits violates partial-volume semantics
(negative values skew the average regardless of the surrounding positive
voxels, breaking the enclosed-lesion-fraction interpretation).
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from vena.common import CropPadSpec, apply_crop_pad
from vena.data.h5.latent_domain.manifest import LATENT_CROP_BOX, LATENT_SPATIAL
from vena.segmentation.config import DerivationConfig
from vena.segmentation.exceptions import SegDerivationError

# Depth-pad base = spatial_compression * _DEPTH_PAD_MULTIPLIER, matching
# PerClassAvgPoolDownsampler (depth_pad_base = 4 * 2 = 8).
_DEPTH_PAD_MULTIPLIER: int = 2


def pool_to_latent(
    prob_img: Tensor,
    cfg: DerivationConfig,
    *,
    crop_spec: CropPadSpec | None = None,
) -> Tensor:
    """Map a ``(2, H, W, D)`` probability image to the latent grid.

    Registration convention mirrors ``masks/tumor_latent`` in the latent H5:

    * Sigmoid / temperature scaling must be applied *before* this call.
    * If ``crop_spec`` is provided the native-space volume is first cropped
      to ``LATENT_CROP_BOX = (192, 224, 192)``; otherwise the caller supplies
      an input already at the crop-box size (the common case when the
      segmenter runs on already-cropped inputs from the latent pipeline).
    * Average pooling with ``kernel_size = stride = cfg.avg_pool_stride``
      (default 4) maps the crop box to the latent grid ``(48, 56, 48)``.

    Parameters
    ----------
    prob_img : Tensor
        Soft probability map in ``[0, 1]``, shape ``(2, H, W, D)``.
        Channel 0 = WT, channel 1 = NETC.  **Sigmoid and temperature scaling
        MUST be applied before calling this function.**
    cfg : DerivationConfig
        Derivation settings.  Uses ``cfg.avg_pool_stride`` and
        ``cfg.latent_grid``.
    crop_spec : CropPadSpec or None, optional
        Per-scan brain-centred crop specification produced by the latent H5
        converter
        (``CropPadSpec(crop_origin=..., native_shape=...,
        target_shape=(192, 224, 192))``).
        When provided, the native-space ``prob_img`` is cropped/padded to
        ``LATENT_CROP_BOX`` before pooling.  When None, ``prob_img`` must
        already be at the crop-box size.

    Returns
    -------
    Tensor
        Soft mask at the latent grid, shape ``(2, *cfg.latent_grid)``
        = ``(2, 48, 56, 48)``.
        Values in ``[0, 1]``; boundary voxels are graded by the enclosed
        lesion fraction (partial-volume integration).

    Raises
    ------
    SegDerivationError
        If ``prob_img`` does not have exactly 2 channels and 4 dimensions,
        or if the output grid does not match ``cfg.latent_grid``.
    """
    if prob_img.ndim != 4 or prob_img.shape[0] != 2:
        raise SegDerivationError(
            f"pool_to_latent expects shape (2, H, W, D); got {tuple(prob_img.shape)}"
        )

    x: Tensor = prob_img  # (2, H, W, D) throughout

    # Step 1: crop to the brain-centred box when a native-space volume is given.
    # apply_crop_pad requires exactly 5-D (B, C, H, W, D); expand and squeeze.
    if crop_spec is not None:
        x = x.unsqueeze(0)  # (1, 2, H, W, D)
        x = apply_crop_pad(x, crop_spec)  # (1, 2, 192, 224, 192)
        x = x.squeeze(0)  # (2, 192, 224, 192)

    # Step 2: depth-pad D to a multiple of avg_pool_stride * _DEPTH_PAD_MULTIPLIER
    # (= 8 at default stride 4), matching PerClassAvgPoolDownsampler convention.
    # For the canonical crop box (D=192), 192 % 8 == 0, so this is a no-op.
    stride = cfg.avg_pool_stride
    depth_pad_base = stride * _DEPTH_PAD_MULTIPLIER
    d = x.shape[-1]
    d_padded = d + (-d % depth_pad_base)
    if d_padded != d:
        x = F.pad(x, (0, d_padded - d))  # pad last (depth) dim only

    # Step 3: avg-pool with kernel = stride.
    # F.avg_pool3d accepts the unbatched 4-D form (C, H, W, D) directly.
    x = F.avg_pool3d(x, kernel_size=stride, stride=stride)  # (2, h, w, d)

    # Validate output grid against cfg.latent_grid (single source of truth).
    expected: tuple[int, int, int] = cfg.latent_grid
    actual: tuple[int, ...] = tuple(x.shape[1:])
    if actual != expected:
        raise SegDerivationError(
            f"pool_to_latent: output grid {actual} != cfg.latent_grid {expected}. "
            f"Ensure prob_img spatial dims are compatible with avg_pool_stride={stride} "
            f"or provide crop_spec to crop to LATENT_CROP_BOX {LATENT_CROP_BOX}."
        )

    # Cross-check against the canonical manifest constant so any future
    # LATENT_SPATIAL drift causes a clear failure here rather than silent misuse.
    assert actual == LATENT_SPATIAL, (
        f"Output grid {actual} diverged from LATENT_SPATIAL {LATENT_SPATIAL}. "
        "Update DerivationConfig.latent_grid to match the served latent H5."
    )

    return x


__all__ = ["pool_to_latent"]

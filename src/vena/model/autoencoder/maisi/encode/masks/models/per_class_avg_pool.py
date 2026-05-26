"""Per-class soft mask downsampler via 3D average pooling.

Each BraTS label is one-hot encoded into its own channel, depth-padded to a
multiple of the spatial compression factor (so the input shares the encoder's
target grid), and then average-pooled by the same factor. The resulting
tensor is a 3-channel ``float32`` map in ``[0, 1]`` where each voxel reports
the fraction of that label inside the corresponding image-space ``4³`` block.

This is the canonical mask conditioning for the latent flow-matching
ControlNet (proposal §3.2 / §5): partial-volume information at the boundary
of the ET ring is preserved, which a hard nearest-neighbour downsample would
discard.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from types import MappingProxyType

from vena.model.autoencoder.maisi.exceptions import ShapeContractError
from vena.model.autoencoder.maisi.preprocessing import (
    DepthPad,
    pad_depth_to_multiple_of,
)

from ..abc_model import AbstractMaskDownsampler
from ..shared.validation import (
    assert_image_space_mask,
    assert_label_codes_subset,
    assert_target_shape,
)


class PerClassAvgPoolDownsampler(AbstractMaskDownsampler):
    """Soft per-label downsampler (NETC, ED, ET) for BraTS tumour masks.

    Parameters
    ----------
    spatial_compression : int
        Pool window size and stride. Defaults to ``4`` (MAISI VAE
        downsampling factor).
    depth_pad_base : int
        Base for depth-axis padding before pooling so the output grid is
        identical to the latent grid. Defaults to ``8`` (twice the
        compression so future encoder revisions still align).
    strict_labels : bool
        If ``True`` (default), the input mask must contain only codes in
        ``{0, 1, 2, 4}``; otherwise the extras are silently ignored.
    """

    name = "per_class_avg_pool"
    output_channels = 3
    output_dtype = torch.float32
    channel_names = ("NETC", "ED", "ET")

    LABEL_CODES = MappingProxyType({"NETC": 1, "ED": 2, "ET": 4})

    def __init__(
        self,
        spatial_compression: int = 4,
        depth_pad_base: int = 8,
        strict_labels: bool = True,
    ) -> None:
        if spatial_compression < 1:
            raise ValueError(f"spatial_compression must be >= 1; got {spatial_compression}")
        if depth_pad_base % spatial_compression != 0:
            raise ValueError(
                f"depth_pad_base ({depth_pad_base}) must be a multiple of "
                f"spatial_compression ({spatial_compression})"
            )
        self.spatial_compression = spatial_compression
        self.depth_pad_base = depth_pad_base
        self.strict_labels = strict_labels

    # ------------------------------------------------------------------ API

    def downsample(
        self,
        mask: torch.Tensor,
        target_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        assert_image_space_mask(mask)
        if self.strict_labels:
            assert_label_codes_subset(mask, self.LABEL_CODES.values())

        # One-hot encode → (B, 3, H, W, D), float32.
        codes = torch.tensor(
            list(self.LABEL_CODES.values()),
            dtype=torch.int64,
            device=mask.device,
        )
        m = mask.to(torch.int64)
        # Broadcast equality across the codes axis: shape (B, 3, H, W, D).
        onehot = (m == codes.view(1, -1, 1, 1, 1)).to(self.output_dtype)

        # Depth-pad so D is divisible by the compression factor.
        onehot, pad = pad_depth_to_multiple_of(onehot, base=self.depth_pad_base)
        self._assert_input_divisible(onehot.shape[-3:], pad)

        k = self.spatial_compression
        pooled = F.avg_pool3d(onehot, kernel_size=k, stride=k)
        assert_target_shape(tuple(pooled.shape[-3:]), tuple(target_shape))
        return pooled

    def to_attrs(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_channels": self.output_channels,
            "channel_names": list(self.channel_names),
            "label_codes": dict(self.LABEL_CODES),
            "spatial_compression": self.spatial_compression,
            "depth_pad_base": self.depth_pad_base,
            "strict_labels": self.strict_labels,
            "method": "one-hot per label, depth-pad to multiple of depth_pad_base, F.avg_pool3d",
        }

    # ------------------------------------------------------------------ helpers

    def _assert_input_divisible(
        self,
        spatial: tuple[int, ...],
        pad: DepthPad,
    ) -> None:
        if any(d % self.spatial_compression != 0 for d in spatial):
            raise ShapeContractError(
                f"after depth-pad, spatial shape {spatial} not divisible by "
                f"spatial_compression={self.spatial_compression} (pad info={pad})"
            )

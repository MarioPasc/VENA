"""Channel-lifting downsampler — learned ``Conv3d(1, C, kernel_size=1)``.

Lifts a single-channel mask (typical: 1-channel WT mask at latent
resolution) to a multi-channel feature map matching the energy of the
4-channel latent modalities (T1pre, T2, FLAIR) in the conditioning
assembler. Without this lift, the WT mask contributes ~1/13 = 7.7 % of the
input energy to the ControlNet's conditioning-embedding first conv, where
the dominant 12 latent channels suppress its signal.

The 2026-06-20 T1C-RFlow comparison (Hypothesis H5) flagged the mask
under-weighting as a candidate cause of the S1 tumour-region failure mode.
This lifter is the reserved-now-used-later infrastructure; S1 keeps
``mask:wt:zero_out`` for warm-start compatibility, and S2/S3 will switch
to ``mask:wt:lift_to_4ch`` in a downstream PR.

References
----------
Park et al. 2019, *Semantic Image Synthesis with Spatially-Adaptive
Normalization*, CVPR (the spatial-modulation precedent for mask channel
lifting); the 1×1×1 conv approach is the lightest realisation that
preserves spatial structure without introducing per-pixel normalisation.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import AbstractDownsampler


class LiftTo4ChDownsampler(AbstractDownsampler):
    """Learned channel-lifting operator via a 1×1×1 convolution.

    Parameters
    ----------
    in_channels : int
        Input channel count. Default ``1`` (single-channel mask).
    out_channels : int
        Output channel count. Default ``4`` (MAISI-V2 latent channels).
    bias : bool
        Whether the convolution has a bias. Default ``False`` to keep the
        lifting purely linear (matches the canonical 1×1 conv recipe in
        SPADE/AdaLN literature).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive; "
                f"got in={in_channels}, out={out_channels}"
            )
        self._out_channels = int(out_channels)
        self.conv = nn.Conv3d(int(in_channels), int(out_channels), kernel_size=1, bias=bool(bias))
        nn.init.kaiming_uniform_(self.conv.weight, a=0)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

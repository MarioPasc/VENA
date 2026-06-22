"""Zero-out downsampler — replaces every voxel with zero.

The conditioning slot is preserved (channel count unchanged) but carries no
information signal. Used by the S1 baseline to keep the WT-mask channel
position byte-compatible with S2/S3 (which switch to ``identity`` or
``lift_to_4ch``) so warm-start from S1 → S2/S3 has no shape mismatch in the
ControlNet's conditioning-embedding first conv.

Reference: 2026-06-20 analysis
(``.claude/notes/changes/decoder_perceptual_loss_s3_analysis_2026-06-20.md``),
§4 — recipe E1.
"""

from __future__ import annotations

import torch

from .base import AbstractDownsampler


class ZeroOutDownsampler(AbstractDownsampler):
    """Stateless operator returning ``torch.zeros_like(x)``.

    Notes
    -----
    No parameters. ``out_channels is None`` (delegates to the kind-based
    default), so the assembler treats the slot as a 1-channel mask carrying
    zeros — identical channel layout to ``mask:wt:identity``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

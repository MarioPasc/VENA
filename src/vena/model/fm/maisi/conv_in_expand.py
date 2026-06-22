"""Channel-expand the MAISI trunk's ``conv_in`` for input-concat conditioning.

S1 v3 (2026-06-22) ships two variants where the MAISI rectified-flow trunk's
first convolution must accept the noisy T1c latent *plus* the concatenated
T1pre / T2 / FLAIR latents — 16 channels (= 4×4) instead of the pretrained
``in_channels=4``. T1C-RFlow (Eidex 2025) does the same thing at
``in_channels=12`` (no T2): the trunk's first conv sees the conditioning
modalities **directly**, not through ControlNet's residual injection.

Mechanism
---------
1. Read the original ``trunk.conv_in.weight`` of shape
   ``(C_out, 4, kT, kH, kW)``.
2. Allocate a new ``nn.Conv3d`` with ``in_channels = new_in_channels`` and the
   same out_channels / kernel / stride / padding / bias setup.
3. Copy the original 4 input-channel slices into the new conv's first 4
   input-channel slices. Zero-fill the remaining ``new_in_channels − 4``.
4. Replace ``trunk.conv_in`` with the new module.

Bit-identical at step 0
-----------------------
With ``zero_init_new=True`` the trunk's output is bit-identical to the
pretrained one when the new input slice is **anything**:
``trunk_expanded(cat([x_old, X], dim=1)) ≡ trunk_original(x_old)``, because the
zero-init weights in channels 4..new annihilate the contribution from X
inside the convolution sum. This is the load-bearing guarantee — it lets the
expanded trunk start training from the MAISI pretrained behaviour and learn
the conditioning channels' weights from zero, analogous to ControlNet's
zero-conv idea applied at the input layer.

Caveats
-------
* ``grad_safe.py`` 's patched ``_forward_oop`` calls ``self.conv_in(x)`` via
  attribute lookup (not a captured reference) — so swapping ``trunk.conv_in``
  after ``make_trunk_grad_safe`` is safe.
* MAISI's pretrained ``conv_in`` carries a bias; the bias is preserved
  verbatim.
* This function mutates the trunk in place (and returns it for chaining).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ConvInExpansionError(RuntimeError):
    """Raised when ``expand_conv_in`` cannot perform the channel expansion."""


def expand_conv_in(
    trunk: nn.Module,
    new_in_channels: int,
    *,
    zero_init_new: bool = True,
) -> nn.Module:
    """Replace ``trunk.conv_in`` with a wider-input ``nn.Conv3d``.

    Parameters
    ----------
    trunk : nn.Module
        A MAISI-style trunk exposing ``trunk.conv_in`` as ``nn.Conv3d``.
    new_in_channels : int
        Target input-channel count. Must be ≥ current ``in_channels``;
        equality is a no-op.
    zero_init_new : bool
        When ``True`` (default), the additional channels' weights are
        zero-initialised — guaranteeing bit-identical output to the original
        trunk at step 0 regardless of what the new inputs carry. When
        ``False``, Kaiming-uniform init is applied to the additional slices.

    Returns
    -------
    nn.Module
        The same trunk instance with its ``conv_in`` replaced.

    Raises
    ------
    ConvInExpansionError
        If ``trunk.conv_in`` is not an ``nn.Conv3d`` or if
        ``new_in_channels`` is smaller than the current ``in_channels``.
    """
    old = getattr(trunk, "conv_in", None)
    if not isinstance(old, nn.Conv3d):
        raise ConvInExpansionError(f"trunk.conv_in must be nn.Conv3d; got {type(old).__name__}")
    old_in = int(old.in_channels)
    if new_in_channels < old_in:
        raise ConvInExpansionError(
            f"new_in_channels ({new_in_channels}) must be ≥ current in_channels ({old_in})"
        )
    if new_in_channels == old_in:
        logger.info("expand_conv_in: noop (in_channels already %d).", old_in)
        return trunk

    new = nn.Conv3d(
        in_channels=new_in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    # Place the new layer on the same device/dtype as the original weights so
    # the trunk stays self-consistent (avoids a downstream ``.to(device)`` step
    # for this lone module).
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        if zero_init_new:
            new.weight.zero_()
        else:
            nn.init.kaiming_uniform_(new.weight, a=5**0.5)
        # Copy the original slices verbatim. Bit-identical at step 0.
        new.weight[:, :old_in].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)

    trunk.conv_in = new
    logger.info(
        "expand_conv_in: trunk.conv_in expanded %d → %d channels (zero_init_new=%s).",
        old_in,
        new_in_channels,
        zero_init_new,
    )
    return trunk

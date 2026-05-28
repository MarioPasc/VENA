"""Make MONAI's MAISI diffusion U-Net safe for trunk fine-tuning.

MONAI's ``DiffusionModelUNetMaisi`` injects the ControlNet residuals into the
trunk activations with two **in-place** adds:

* ``down_block_res_sample += down_block_additional_residual`` (in
  ``_apply_down_blocks``)
* ``h += mid_block_additional_residual`` (in ``forward``)

With a *frozen* trunk those activations carry no gradient, so the in-place write
is harmless and the canonical (ControlNet-only) recipe trains fine. Once the
trunk is *unfrozen* (the unfrozen-trunk ablation, cf. TumorFlow) autograd needs
the un-mutated output of the preceding residual add and raises::

    RuntimeError: one of the variables needed for gradient computation has been
    modified by an inplace operation: [... 1, 256, 8, 8, 5 ...], which is output
    0 of AddBackward0, is at version 1; expected version 0 instead.

We do **not** edit the installed MONAI package. Instead we rebind the two
methods on the *instance* with out-of-place equivalents (monkeypatch-in-adapter,
per ``external-deps.md``). The arithmetic is identical — only ``x += y`` becomes
``x = x + y`` — so the frozen-trunk numerics are unchanged; this is purely an
autograd-graph fix that allows gradients to flow back into the trunk.

TumorFlow trains trunk + ControlNet jointly using the **stock** (non-MAISI)
``monai.networks.nets.DiffusionModelUNet``, whose residual injection is already
out-of-place — which is plausibly why they never hit this.
"""

from __future__ import annotations

import logging
import types
from typing import Any

import torch
from monai.utils import convert_to_tensor

logger = logging.getLogger(__name__)


def _apply_down_blocks_oop(
    self: Any,
    h: torch.Tensor,
    emb: torch.Tensor,
    context: torch.Tensor | None,
    down_block_additional_residuals: tuple[torch.Tensor, ...] | None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Out-of-place clone of ``DiffusionModelUNetMaisi._apply_down_blocks``."""
    if context is not None and self.with_conditioning is False:
        raise ValueError("model should have with_conditioning = True if context is provided")
    down_block_res_samples: list[torch.Tensor] = [h]
    for downsample_block in self.down_blocks:
        h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
        down_block_res_samples.extend(res_samples)

    if down_block_additional_residuals is not None:
        new_down_block_res_samples: list[torch.Tensor] = []
        for sample, residual in zip(down_block_res_samples, down_block_additional_residuals):
            new_down_block_res_samples.append(sample + residual)  # was ``sample += residual``
        down_block_res_samples = new_down_block_res_samples
    return h, down_block_res_samples


def _forward_oop(
    self: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor | None = None,
    class_labels: torch.Tensor | None = None,
    down_block_additional_residuals: tuple[torch.Tensor, ...] | None = None,
    mid_block_additional_residual: torch.Tensor | None = None,
    top_region_index_tensor: torch.Tensor | None = None,
    bottom_region_index_tensor: torch.Tensor | None = None,
    spacing_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Out-of-place clone of ``DiffusionModelUNetMaisi.forward``."""
    emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
    emb = self._get_input_embeddings(
        emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor
    )
    h = self.conv_in(x)
    h, updated_down_block_res_samples = self._apply_down_blocks(
        h, emb, context, down_block_additional_residuals
    )
    h = self.middle_block(h, emb, context)

    if mid_block_additional_residual is not None:
        h = h + mid_block_additional_residual  # was ``h += mid_block_additional_residual``

    h = self._apply_up_blocks(h, emb, context, updated_down_block_res_samples)
    h = self.out(h)
    h_tensor: torch.Tensor = convert_to_tensor(h)
    return h_tensor


def make_trunk_grad_safe(model: torch.nn.Module) -> torch.nn.Module:
    """Rebind the trunk's residual-injection methods to out-of-place versions.

    Idempotent and instance-local: rebinds ``_apply_down_blocks`` and ``forward``
    as bound methods on ``model`` only. Other instances and the MONAI class are
    untouched. Call this when the trunk is trainable.

    Parameters
    ----------
    model : torch.nn.Module
        A ``DiffusionModelUNetMaisi`` instance.

    Returns
    -------
    torch.nn.Module
        The same instance, patched.
    """
    model._apply_down_blocks = types.MethodType(_apply_down_blocks_oop, model)
    model.forward = types.MethodType(_forward_oop, model)
    logger.info(
        "Trunk patched grad-safe: out-of-place ControlNet residual adds (down-block + mid-block)."
    )
    return model

"""MAISI-style ControlNet adapter wrapping ``monai`` 's ``ControlNetMaisi``.

The implementation follows the original ControlNet recipe
(Zhang et al. *ControlNet*, ICCV 2023):

1. **Encoder warm-start**: deep-copy the trunk's encoder (``conv_in``,
   ``time_embed``, ``class_embedding``, ``down_blocks``, ``middle_block``) into
   the ControlNet via ``load_state_dict(strict=False)``.
2. **Zero output projections**: ``controlnet_down_blocks.*`` and
   ``controlnet_mid_block.*`` are zeroed so that at step 0,
   ``trunk_with_controlnet(...) == trunk(...)``.

After step 0, gradients flow only through the ControlNet — the trunk stays
frozen (handled by the caller via :class:`vena.model.fm.maisi.TrunkHandle`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import AbstractControlNet

logger = logging.getLogger(__name__)


class MaisiControlNet(AbstractControlNet):
    """Concrete ControlNet wrapping ``ControlNetMaisi``.

    Parameters
    ----------
    conditioning_in_channels : int
        Channel count of the conditioning tensor at the input. Must equal the
        sum over the :class:`ConditioningAssembler` 's input specs.
    arch_overrides : dict | None
        Per-call overrides applied on top of the JSON config — e.g. swapping
        ``conditioning_embedding_num_channels``.
    arch_config : Path | None
        Optional override for the architecture-kwargs JSON. Defaults to the
        bundled ``configs/controlnet_rflow.json``.
    """

    DEFAULT_ARCH_CONFIG = Path(__file__).parent / "configs" / "controlnet_rflow.json"

    def __init__(
        self,
        conditioning_in_channels: int,
        arch_overrides: dict[str, Any] | None = None,
        arch_config: Path | str | None = None,
        init_from_trunk_enabled: bool = True,
    ) -> None:
        """Construct a MAISI ControlNet branch.

        ``init_from_trunk_enabled=True`` (default) preserves the canonical
        ControlNet recipe: the LightningModule subsequently calls
        :meth:`init_from_trunk` to shape-filter-copy the trunk's encoder
        weights. S1 v3 Variant B sets this flag to ``False`` because the
        ControlNet now consumes a 3-channel mask, not the modality latents —
        the trunk's pretrained encoder weights are no longer the right
        warm-start for the cond_embedding's downstream blocks (they still
        get copied for the block params themselves; only the high-level
        "should we re-init from trunk?" gate is exposed). Even with the
        flag flipped, :meth:`zero_init_output_projections` is still run by
        the caller — without it the residual injection would not be
        additive-from-zero.
        """
        super().__init__()
        self.conditioning_in_channels = int(conditioning_in_channels)
        self.init_from_trunk_enabled = bool(init_from_trunk_enabled)

        arch_path = Path(arch_config) if arch_config is not None else self.DEFAULT_ARCH_CONFIG
        with arch_path.open("r") as f:
            raw = json.load(f)
        arch_kwargs: dict[str, Any] = {k: v for k, v in raw.items() if not k.startswith("_")}
        arch_kwargs["conditioning_embedding_in_channels"] = self.conditioning_in_channels
        if arch_overrides:
            arch_kwargs.update(arch_overrides)
        self._arch_kwargs = arch_kwargs

        from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi

        self.net = ControlNetMaisi(**arch_kwargs)
        self.zero_init_output_projections()
        # Scale-ramped zero-init (2026-06-20 analysis §4a, recipe E1):
        # multiplied into every down-block residual and the mid-block residual
        # in :meth:`forward`. Driven by
        # :class:`vena.model.fm.lightning.callbacks.OutputScaleRampCallback`,
        # which fills it with a sigmoid ramp from ~0 → 1 over ``ramp_steps``.
        # Default 1.0 preserves byte-identical behaviour when the callback is
        # not registered (S1 retired runs, any tooling that builds the
        # ControlNet outside Lightning). Non-persistent: the ramp formula is
        # deterministic in ``global_step`` so we recompute on resume rather
        # than checkpointing a value that may diverge from the formula.
        self.register_buffer(
            "output_scale",
            torch.tensor(1.0),
            persistent=False,
        )
        logger.info(
            "MaisiControlNet built: cond_in=%d arch_keys=%s output_scale=1.0",
            self.conditioning_in_channels,
            list(arch_kwargs.keys()),
        )

    @property
    def arch_kwargs(self) -> dict[str, Any]:
        return dict(self._arch_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        down_block_res_samples, mid_block_res_sample = self.net(
            x=x,
            timesteps=timesteps,
            controlnet_cond=controlnet_cond,
            class_labels=class_labels,
        )
        # Apply the scale-ramped zero-init (default 1.0 = byte-identical).
        # The scalar is a non-grad buffer so autograd treats it as a constant
        # at each step; the ControlNet parameters' gradients are unaffected.
        # Sits upstream of ``vena.model.fm.maisi.grad_safe`` 's out-of-place
        # residual-add patch in the trunk's forward — no interaction with
        # gradient checkpointing.
        scale = self.output_scale
        down_block_res_samples = [t * scale for t in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * scale
        return down_block_res_samples, mid_block_res_sample

    def init_from_trunk(self, trunk_state_dict: dict) -> None:
        """Deep-copy the matching encoder + mid-block weights from the trunk.

        We perform a shape-filtered ``load_state_dict``: only keys that exist
        on both sides *and* whose tensors have matching shapes are copied.

        The remaining classes are tolerated:

        * **Trunk-only keys** (``up_blocks``, ``out``, ``conv_out``,
          ``spacing_layer``, …): the ControlNet does not own them.
        * **ControlNet-only keys** (``controlnet_cond_embedding``,
          ``controlnet_down_blocks``, ``controlnet_mid_block``): the trunk
          does not own them; they retain the constructor defaults until
          :meth:`zero_init_output_projections` zeroes the output projections.
        * **Shape-mismatched keys**: most notably
          ``*.time_emb_proj.weight``. The MAISI trunk has
          ``include_spacing_input=True``, which doubles the effective
          time-embedding width (time + spacing). The MAISI ControlNet has no
          spacing-input pathway and therefore its ``time_emb_proj`` is
          half-width. Skipping those keys is acceptable: the ControlNet
          paper's warm-start prescription is the *encoder convolutions*; the
          time projections are a small, fast-to-learn linear map.
        * **trunk.conv_in shape mismatch** (S1 v3): when the trunk's
          ``conv_in`` was expanded from 4 → 16 channels via
          :func:`vena.model.fm.maisi.conv_in_expand.expand_conv_in`, its
          shape no longer matches the ControlNet's own 4-channel
          ``conv_in``. The mismatch is silently skipped by this method's
          shape filter — the ControlNet's ``conv_in`` keeps the *original*
          4-channel form (its forward input ``x`` is always the noisy T1c
          latent, never the concat). This is the intended behaviour.

        S1 v3 Variant B disables this call entirely (via
        ``init_from_trunk_enabled=False`` on the constructor) because the
        new 3-channel mask cond_embedding has no useful warm-start from the
        trunk's 4-channel-input encoder.
        """
        if not self.init_from_trunk_enabled:
            logger.info(
                "ControlNet init_from_trunk: SKIPPED (init_from_trunk_enabled=False); "
                "encoder remains at MONAI's constructor init."
            )
            return
        own_sd = self.net.state_dict()
        copyable: dict[str, Any] = {}
        shape_mismatch: list[str] = []
        for k, v in trunk_state_dict.items():
            if k not in own_sd:
                continue
            if own_sd[k].shape == v.shape:
                copyable[k] = v
            else:
                shape_mismatch.append(f"{k}: trunk={tuple(v.shape)} cn={tuple(own_sd[k].shape)}")
        missing, unexpected = self.net.load_state_dict(copyable, strict=False)
        logger.info(
            "ControlNet init_from_trunk: copied=%d shape_mismatch=%d "
            "missing(cn-only-after-copy)=%d unexpected=%d",
            len(copyable),
            len(shape_mismatch),
            len(missing),
            len(unexpected),
        )
        if shape_mismatch:
            logger.info(
                "  shape-mismatched keys (showing first 3): %s",
                shape_mismatch[:3],
            )

    def zero_init_output_projections(self) -> None:
        with torch.no_grad():
            for name, p in self.net.named_parameters():
                if name.startswith("controlnet_down_blocks.") or name.startswith(
                    "controlnet_mid_block."
                ):
                    nn.init.zeros_(p)

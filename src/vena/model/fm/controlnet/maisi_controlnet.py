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

    DEFAULT_ARCH_CONFIG = (
        Path(__file__).parent / "configs" / "controlnet_rflow.json"
    )

    def __init__(
        self,
        conditioning_in_channels: int,
        arch_overrides: dict[str, Any] | None = None,
        arch_config: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.conditioning_in_channels = int(conditioning_in_channels)

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
        logger.info(
            "MaisiControlNet built: cond_in=%d arch_keys=%s",
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
        return self.net(
            x=x,
            timesteps=timesteps,
            controlnet_cond=controlnet_cond,
            class_labels=class_labels,
        )

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
        """
        own_sd = self.net.state_dict()
        copyable: dict[str, Any] = {}
        shape_mismatch: list[str] = []
        for k, v in trunk_state_dict.items():
            if k not in own_sd:
                continue
            if own_sd[k].shape == v.shape:
                copyable[k] = v
            else:
                shape_mismatch.append(
                    f"{k}: trunk={tuple(v.shape)} cn={tuple(own_sd[k].shape)}"
                )
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

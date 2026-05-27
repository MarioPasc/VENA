"""Abstract base class for the trainable ControlNet branch.

The ControlNet branch is the only trainable network in VENA's default
configuration (proposal §4.3). Concrete subclasses must:

1. Expose ``conditioning_in_channels`` — the channel count the
   :class:`ConditioningAssembler` is expected to produce.
2. Provide a forward returning ``(down_block_res_samples, mid_block_res_sample)``
   compatible with the MAISI trunk's ``down_block_additional_residuals`` and
   ``mid_block_additional_residual`` kwargs (proposal §4.3).
3. Support :meth:`init_from_trunk` and :meth:`zero_init_output_projections`,
   matching the original ControlNet initialisation prescription (Zhang et al.,
   ICCV 2023; encoder-half deep copy + zero output projections so that the
   augmented forward equals the pretrained trunk at step 0).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractControlNet(nn.Module, ABC):
    """Trainable ControlNet adapter on top of a frozen FM trunk."""

    #: Channel count of the conditioning tensor produced by the assembler.
    conditioning_in_channels: int

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the ControlNet branch.

        Parameters
        ----------
        x : Tensor
            Noisy latent ``(B, C_lat, H, W, D)`` — same as the trunk input.
        timesteps : Tensor
            Integer timesteps ``(B,)`` on the same device as ``x``.
        controlnet_cond : Tensor
            Conditioning tensor ``(B, conditioning_in_channels, H, W, D)``
            produced by :class:`ConditioningAssembler`.
        class_labels : Tensor | None
            Optional modality / class token ``(B,)``.

        Returns
        -------
        tuple[list[Tensor], Tensor]
            ``(down_block_res_samples, mid_block_res_sample)`` — feed directly
            into :meth:`DiffusionModelUNetMaisi.forward` as
            ``down_block_additional_residuals`` and
            ``mid_block_additional_residual``.
        """

    @abstractmethod
    def init_from_trunk(self, trunk_state_dict: dict) -> None:
        """Copy matching encoder + mid-block weights from the trunk.

        ``strict=False`` style: keys that do not exist in the ControlNet are
        ignored, and the ControlNet's own conditioning-embedding layers (which
        the trunk does not have) are left at their constructor defaults.
        """

    @abstractmethod
    def zero_init_output_projections(self) -> None:
        """Zero-initialise the output projections (zero convolutions).

        After this call, the augmented forward must equal the pretrained
        trunk's forward at step 0, regardless of the conditioning tensor.
        """

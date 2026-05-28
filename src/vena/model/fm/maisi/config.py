"""Pydantic config for the frozen MAISI-V2 rectified-flow trunk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field


class TrunkConfig(BaseModel):
    """User-facing config for :func:`load_trunk`.

    Attributes
    ----------
    checkpoint : Path
        Absolute path to ``diff_unet_3d_rflow-mr.pt``.
    arch_json : Path | None
        Optional override for the architecture-kwargs JSON.
    arch_overrides : dict
        Per-call overrides applied on top of the JSON config.
    class_token : int
        MAISI modality class id. ``9`` corresponds to ``mri_t1`` and is the
        target modality for the synthesised T1c (proposal §4.2).
    spacing_mm : tuple of float
        Voxel spacing in millimetres for the trunk's spacing-conditioning input.
        Default ``(1.0, 1.0, 1.0)`` matches the 1 mm isotropic preprocessing.
    trainable : bool
        When ``True`` (project default) the trunk is unfrozen and fine-tuned
        jointly with the ControlNet, motivated by TumorFlow's joint training of
        trunk + ControlNet for an out-of-distribution latent target. Set to
        ``False`` to recover the canonical frozen-backbone ControlNet recipe
        (proposal §4.2), where only the ControlNet is optimised.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint: Path
    arch_json: Path | None = None
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    class_token: int = 9
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    trainable: bool = True

    def make_class_labels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a ``(B,)`` long tensor of ``class_token`` for every sample."""
        return torch.full(
            (batch_size,), int(self.class_token), dtype=torch.long, device=device
        )

    def make_spacing_tensor(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a ``(B, 3)`` float tensor of ``spacing_mm`` for every sample."""
        return torch.tensor(self.spacing_mm, dtype=torch.float32, device=device).expand(
            batch_size, 3
        )

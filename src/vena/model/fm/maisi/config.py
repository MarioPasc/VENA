"""Pydantic config for the frozen MAISI-V2 rectified-flow trunk."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    regime : Literal['fft', 'peft']
        How a trainable trunk is parameterised. ``'fft'`` (default) is the
        full-fine-tune path that updates every trunk tensor. ``'peft'`` routes
        through :mod:`vena.model.fm.maisi.peft` to inject adapter tensors
        (LoRA / IA3 / DoRA / ...) on top of the frozen backbone; the
        ``peft`` block then selects the variant and its parameters.
    peft : dict | None
        PEFT variant + parameters block. Required when ``regime == 'peft'``;
        must be absent or ``None`` otherwise. The expected shape is
        ``{"variant": "<name>", "params": {...}}`` where the params schema is
        owned by each variant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint: Path
    arch_json: Path | None = None
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    class_token: int = 9
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    trainable: bool = True
    regime: Literal["fft", "peft"] = "fft"
    peft: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_regime_peft(self) -> TrunkConfig:
        if self.regime == "peft":
            if not self.trainable:
                raise ValueError(
                    "trunk.regime='peft' requires trunk.trainable=true (PEFT on a "
                    "fully frozen trunk has no effect)"
                )
            if not self.peft or "variant" not in self.peft:
                raise ValueError(
                    "trunk.regime='peft' requires a non-empty peft block of the form "
                    "{variant: <name>, params: {...}}"
                )
        else:  # regime == 'fft'
            if self.peft is not None:
                raise ValueError(
                    "trunk.peft must be null when trunk.regime='fft' (got "
                    f"{self.peft!r}); set regime='peft' to enable adapter training"
                )
        return self

    def make_class_labels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a ``(B,)`` long tensor of ``class_token`` for every sample."""
        return torch.full((batch_size,), int(self.class_token), dtype=torch.long, device=device)

    def make_spacing_tensor(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a ``(B, 3)`` float tensor of ``spacing_mm`` for every sample."""
        return torch.tensor(self.spacing_mm, dtype=torch.float32, device=device).expand(
            batch_size, 3
        )

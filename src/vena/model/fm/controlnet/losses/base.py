"""Abstract loss + composite for VENA flow-matching training.

The :class:`CompositeLoss` is the only object the LightningModule talks to.
It owns:

* The list of concrete loss terms (CFM, contrastive, reconstruction, ...).
* The corresponding scalar weights.
* The ``requires_perturbed_pass`` property — tells the Lightning module
  whether a second ControlNet forward (with the mask zeroed) must also be
  computed for this batch. S1 sets this to ``False``; S2/S3 set it to ``True``.

Curriculum stages activate different *subsets* of terms — but the public
forward signature is the same for every stage. The Lightning module never
branches on stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LossInputs:
    """Everything a loss term may need at one training step.

    Attributes
    ----------
    x_clean : Tensor
        Target clean latent ``z_t1c`` of shape ``(B, C, H, W, D)``.
    noise : Tensor
        Gaussian noise ``x0`` of the same shape.
    x_t : Tensor
        Interpolated noisy latent.
    timesteps : Tensor
        Integer timesteps ``(B,)``.
    u_target : Tensor
        Rectified-flow target ``x_clean - noise``.
    v_orig : Tensor
        Trunk output for the *original* (unperturbed) conditioning.
    v_perturb : Tensor | None
        Trunk output for the *perturbed* (mask-zeroed) conditioning. ``None``
        when the composite did not require a second pass (S1).
    m_wt : Tensor | None
        Whole-tumour mask in latent space, shape ``(B, 1, h, w, d)``, binary.
        Used by S2 contrastive and S3 reconstruction. ``None`` for S1.
    m_bg : Tensor | None
        Dilated-complement background mask. Same shape as ``m_wt``.
    """

    x_clean: torch.Tensor
    noise: torch.Tensor
    x_t: torch.Tensor
    timesteps: torch.Tensor
    u_target: torch.Tensor
    v_orig: torch.Tensor
    v_perturb: torch.Tensor | None = None
    m_wt: torch.Tensor | None = None
    m_bg: torch.Tensor | None = None


class AbstractFMLoss(nn.Module, ABC):
    """One curriculum term."""

    @abstractmethod
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        """Return a scalar loss for one batch."""


class CompositeLoss(nn.Module):
    """Weighted sum of concrete loss terms.

    Parameters
    ----------
    terms : dict[str, AbstractFMLoss]
        Named loss modules; the key is used in the per-term log dict.
    weights : dict[str, float]
        Scalar weights; keys must equal ``terms``' keys.
    requires_perturbed_pass : bool
        Whether the composite needs the perturbed ControlNet pass. The
        Lightning module reads this to decide whether to recompute the trunk
        with a zeroed-mask conditioning.
    stage : str
        Human-readable stage label (``"S1"``, ``"S2"``, ``"S3"``) for logging.
    """

    def __init__(
        self,
        terms: dict[str, AbstractFMLoss],
        weights: dict[str, float],
        requires_perturbed_pass: bool,
        stage: str,
    ) -> None:
        super().__init__()
        if set(terms) != set(weights):
            raise ValueError(
                f"terms and weights must share keys; got terms={list(terms)} "
                f"weights={list(weights)}"
            )
        self.terms = nn.ModuleDict(terms)
        self.weights: dict[str, float] = dict(weights)
        self.requires_perturbed_pass = bool(requires_perturbed_pass)
        self.stage = str(stage)

    def forward(self, inputs: LossInputs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_term: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=inputs.x_clean.device, dtype=inputs.x_clean.dtype)
        for name, term in self.terms.items():
            value = term(inputs)
            per_term[name] = value.detach()
            total = total + self.weights[name] * value
        per_term["total"] = total.detach()
        return total, per_term

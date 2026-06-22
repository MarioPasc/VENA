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

from .schedule import StaticWeight, WeightSchedule


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
        when the composite did not require a second pass. The v0.4 contrastive
        does not need this; field retained for code paths (CFG diagnostic,
        ablation harnesses) that still want it.
    m_wt : Tensor | None
        Whole-tumour mask in latent space, shape ``(B, 1, h, w, d)``, binary.
        Read from ``masks/tumor_latent`` by ``LatentH5Dataset``.
    m_bg : Tensor | None
        Dilated-complement background mask (``1 − dilate3(m_wt)``). Kept for
        legacy callers; the v0.4 contrastive uses :attr:`m_brain` instead.
    m_brain : Tensor | None
        Binary brain mask in latent space, shape ``(B, 1, h, w, d)``. Read
        from ``masks/brain_latent`` (max-pool-4 of image-domain ``masks/brain``,
        encoded by ``vena-encode-brain-to-latent``). Required by the v0.4
        contrastive when any term uses ``region ∈ {brain, healthy, background}``.
    m_tumor : Tensor | None
        Soft per-class tumour mask, shape ``(B, 3, h, w, d)``. Channels are
        ``[NETC (label 1), ED (label 2), ET (label 4)]`` per the BraTS21
        convention; produced by ``masks/tumor_latent`` in the latent H5
        (avg-pooled from image-domain one-hot). Consumed by the S1 v3
        region-weighted L1 loss to apply per-sub-region weights. ``None``
        on legacy callers that pre-date v3.
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
    m_brain: torch.Tensor | None = None
    m_tumor: torch.Tensor | None = None


class AbstractFMLoss(nn.Module, ABC):
    """One curriculum term.

    Subclasses may attach diagnostic scalars to ``self._aux`` after each
    forward; :class:`CompositeLoss` merges them into the returned ``per_term``
    dict under namespaced keys (e.g. ``"contrastive/delta_abs_mean_in"``) so
    the LightningModule logs them through the existing ``train/*`` path
    without a signature change.
    """

    def __init__(self) -> None:
        super().__init__()
        self._aux: dict[str, torch.Tensor] = {}

    @abstractmethod
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        """Return a scalar loss for one batch."""

    def aux(self) -> dict[str, torch.Tensor]:
        """Return diagnostic scalars from the most recent ``forward`` call.

        Default: empty dict. Loss terms with useful side-channel diagnostics
        (e.g. the contrastive |Δ| statistics, cap-hit fractions) populate
        ``self._aux`` inside ``forward`` and return it here. The values must be
        detached scalar tensors.
        """
        return self._aux


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
        weights: dict[str, float | WeightSchedule],
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
        # Coerce raw floats into ``StaticWeight`` schedules so the inner loop is
        # uniform. The legacy ``.weights`` dict is kept as a *snapshot of the
        # initial constants* for back-compat with existing tests that read it;
        # the live values come from ``self.schedules[name].at(step, total)``.
        self.schedules: dict[str, WeightSchedule] = {
            name: (w if isinstance(w, WeightSchedule) else StaticWeight(float(w)))
            for name, w in weights.items()
        }
        self.weights: dict[str, float] = {
            name: sched.at(None, None) for name, sched in self.schedules.items()
        }
        self.requires_perturbed_pass = bool(requires_perturbed_pass)
        self.stage = str(stage)

    def forward(
        self,
        inputs: LossInputs,
        *,
        global_step: int | None = None,
        total_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_term: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=inputs.x_clean.device, dtype=inputs.x_clean.dtype)
        for name, term in self.terms.items():
            value = term(inputs)
            per_term[name] = value.detach()
            w = self.schedules[name].at(global_step, total_steps)
            per_term[f"{name}_weight"] = torch.tensor(w, device=value.device, dtype=value.dtype)
            total = total + w * value
            for aux_key, aux_val in term.aux().items():
                per_term[f"{name}/{aux_key}"] = aux_val.detach()
        per_term["total"] = total.detach()
        return total, per_term

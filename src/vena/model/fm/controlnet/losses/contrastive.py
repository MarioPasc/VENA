r"""Region-weighted CFM residual loss (proposal §5.3 v0.4 — 2026-06-09 overhaul).

Replaces the mask-sensitivity ``|v_orig − v_perturb|^p`` formulation (v0.3,
shipped 2026-05-30) that traded synthesis quality for attribution and saturated
its ROI cap after a few thousand steps. The new formulation is a per-sample
sum of Lp residuals between the predicted velocity and the FM target,
restricted to a configurable list of latent-space regions:

.. math::

    \mathcal{L}_{\text{contrastive}}^{(\theta)} \;=\;
        \sum_{\tau \in \text{terms}} w_\tau \cdot \mathcal{L}_\tau^{(p_\tau)}
    \qquad
    \mathcal{L}_\tau^{(p)} \;=\;
        \tfrac{1}{|m_\tau| \cdot C} \sum_{i \in m_\tau} \sum_{c=1}^{C}
            \big| v_{\text{pred}}^{(c)}(i) - u_{\text{target}}^{(c)}(i) \big|^{p}

Each term is one of (binary masks, broadcast over the ``C=4`` latent channels):

==========  =======================
``region``  formula
==========  =======================
``wt``      :math:`m_{\text{wt}}`
``brain``   :math:`m_{\text{brain}}`
``healthy`` :math:`m_{\text{brain}} \wedge \neg m_{\text{wt}}`
``background`` :math:`\neg m_{\text{brain}}`
``full``    :math:`\mathbf{1}` (whole volume)
==========  =======================

The doc-default of CHANGE 2 (single ``healthy`` term, ``p=2``) collapses to
the recipe in the design note, but the YAML now lets us mix terms (e.g. ``wt``
with ``p=1`` and ``healthy`` with ``p=3``) without code changes — see the
``loss.contrastive`` block in ``routines/fm/train/configs/runs/picasso_s2_*``.

Notes
-----
* Operates entirely on latent-space velocity fields; no VAE decode.
* Every term independently ``clamp_min``s its denominator so an empty region
  contributes 0, not NaN.
* The cached per-sample tensor returned by :meth:`per_sample` is the sum of
  ``w_τ × term_per_sample_τ`` — used by the LightningModule's per-cohort
  contrastive breakdown wired in ``module.py:438+``.
* ``requires_perturbed_pass`` is **False** for this loss; the v0.3 perturb-pass
  machinery now exclusively serves the classifier-free-guidance training-time
  dropout path (``training.conditioning_dropout_p`` in the YAML).

DEPRECATION NOTE — the v0.3 schema keys ``lambda_roi``, ``lambda_bg``, ``delta``,
``p_t``, ``p_b`` are no longer accepted. The same applies to the aux keys
``delta_abs_mean_in/out``, ``roi_cap_hit_frac``, ``bg_cap_hit_frac`` which the
old loss emitted to CSV — those columns simply stop appearing in new training
runs' ``train_step.csv`` / ``train_epoch.csv``.

References
----------
* Chartsias *et al.*, *Multimodal MR synthesis via modality-invariant latent
  representation*, IEEE TMI 2018 — region-weighted MRI synthesis loss.
* Konukoglu *et al.*, *Unsupervised lesion detection via image restoration
  with a normative prior*, MedIA 2021 — ROI-restricted reconstruction error.
* Design note: ``.claude/notes/changes/2026-06-09_training-regime-overhaul.md``
  CHANGE 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .base import AbstractFMLoss, LossInputs

_RegionKind = Literal["wt", "brain", "healthy", "background", "full"]
_REGION_KINDS: tuple[_RegionKind, ...] = ("wt", "brain", "healthy", "background", "full")
# Regions that require ``m_brain`` (image-domain ``masks/brain`` max-pool-encoded
# to latent space by ``vena-encode-brain-to-latent``).
_REGIONS_NEEDING_BRAIN: frozenset[_RegionKind] = frozenset({"brain", "healthy", "background"})


@dataclass(frozen=True)
class RegionTerm:
    """One ``(region, p, weight)`` triple inside the contrastive composite.

    Parameters
    ----------
    name : str
        Human-readable label used in CSV column names (``term_<name>_lp_mean``).
        Recommended convention: same as ``region`` unless multiple terms share
        a region with different ``p`` (e.g. ``name='wt_l1'`` for ``region='wt',
        p=1.0``).
    region : str
        One of ``{wt, brain, healthy, background, full}``.
    p : float
        Lp exponent, strictly positive.
    weight : float
        Per-term scalar inside the contrastive bucket. The outer
        ``loss.contrastive.weight`` (a separate scalar, applied by
        :class:`CompositeLoss`) multiplies the *sum* of these.
    """

    name: str
    region: _RegionKind
    p: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.region not in _REGION_KINDS:
            raise ValueError(
                f"RegionTerm.region must be one of {_REGION_KINDS}; got {self.region!r}"
            )
        if self.p <= 0.0:
            raise ValueError(f"RegionTerm.p must be > 0; got {self.p}")
        if not self.name:
            raise ValueError("RegionTerm.name must be a non-empty string")


def _resolve_region(
    region: _RegionKind,
    m_wt: torch.Tensor,
    m_brain: torch.Tensor | None,
) -> torch.Tensor:
    """Materialise a binary mask for ``region`` on the device/dtype of ``m_wt``.

    Returns a tensor of the same shape as ``m_wt`` (``(B, 1, h, w, d)``) with
    values in ``{0.0, 1.0}``.
    """
    if region == "full":
        return torch.ones_like(m_wt)
    if region == "wt":
        return m_wt
    if region == "brain":
        assert m_brain is not None
        return m_brain
    if region == "healthy":
        assert m_brain is not None
        return m_brain * (1.0 - m_wt)
    if region == "background":
        assert m_brain is not None
        return 1.0 - m_brain
    # Unreachable; validated in RegionTerm.__post_init__.
    raise ValueError(f"unknown region {region!r}")


def _masked_lp_per_sample(
    residual: torch.Tensor,
    region_mask: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """Compute per-sample mean of ``|residual|^p`` inside ``region_mask``.

    Parameters
    ----------
    residual : Tensor
        ``(B, C, h, w, d)`` — already ``.abs()``-ed if appropriate.
    region_mask : Tensor
        ``(B, 1, h, w, d)`` — binary, same dtype as ``residual``.
    p : float
        Lp exponent.

    Returns
    -------
    Tensor of shape ``(B,)``. Empty regions contribute 0 via ``clamp_min(1)``
    on the denominator.
    """
    C = float(residual.shape[1])
    weighted = residual.pow(p) * region_mask
    num = weighted.flatten(1).sum(dim=1)
    den = (region_mask.flatten(1).sum(dim=1) * C).clamp_min(1.0)
    return num / den


class ContrastiveTumourLoss(AbstractFMLoss):
    """Region-weighted CFM residual (proposal §5.3 v0.4).

    Parameters
    ----------
    terms : list[RegionTerm]
        Ordered list of region terms; each contributes
        ``weight × mean_{region}(|v_pred − u_target|^p)`` to the per-sample
        total. At least one term required.

    Notes
    -----
    The class name is preserved (not renamed to ``RegionalLpLoss``) because
    downstream plumbing — ``CompositeLoss``'s ``terms["contrastive"]`` key,
    ``train_csv._COHORT_PREFIXES``, the per-cohort logging path in
    ``FMLightningModule.training_step`` — keys off the ``"contrastive"`` name
    set by :func:`build_loss`.
    """

    def __init__(self, terms: list[RegionTerm]) -> None:
        super().__init__()
        if not terms:
            raise ValueError(
                "ContrastiveTumourLoss requires at least one RegionTerm; "
                "an empty list would make the loss a constant zero."
            )
        names = [t.name for t in terms]
        if len(set(names)) != len(names):
            raise ValueError(
                f"RegionTerm names must be unique within a single contrastive "
                f"loss; got duplicates in {names}"
            )
        self.terms: tuple[RegionTerm, ...] = tuple(terms)
        self._needs_brain = any(t.region in _REGIONS_NEEDING_BRAIN for t in terms)

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        if inputs.v_orig is None:
            raise ValueError("ContrastiveTumourLoss requires v_orig (predicted velocity).")
        if inputs.u_target is None:
            raise ValueError("ContrastiveTumourLoss requires u_target (FM target velocity).")
        if inputs.m_wt is None:
            raise ValueError(
                "ContrastiveTumourLoss requires m_wt in LossInputs; the LightningModule "
                "must populate it from the latent H5's masks/tumor_latent."
            )
        if self._needs_brain and inputs.m_brain is None:
            raise ValueError(
                "ContrastiveTumourLoss has a region term in "
                "{brain, healthy, background} but m_brain is None. Re-encode the "
                "latent H5 with `vena-encode-brain-to-latent` so masks/brain_latent "
                "is present, or use a terms list restricted to {wt, full}."
            )

        residual = (inputs.v_orig - inputs.u_target).abs()  # (B, C, h, w, d)
        m_wt = inputs.m_wt.to(residual.dtype)
        m_brain = inputs.m_brain.to(residual.dtype) if inputs.m_brain is not None else None

        # Accumulate per-sample sum across terms; cache per-term diagnostics.
        per_sample_total = torch.zeros(
            residual.shape[0], device=residual.device, dtype=residual.dtype
        )
        aux: dict[str, torch.Tensor] = {}
        for term in self.terms:
            region_mask = _resolve_region(term.region, m_wt, m_brain)
            term_ps = _masked_lp_per_sample(residual, region_mask, term.p)
            per_sample_total = per_sample_total + float(term.weight) * term_ps
            with torch.no_grad():
                aux[f"term_{term.name}_lp_mean"] = term_ps.mean().detach()
                voxel_frac = region_mask.flatten(1).sum(dim=1) / float(
                    region_mask.shape[1]
                    * region_mask.shape[2]
                    * region_mask.shape[3]
                    * region_mask.shape[4]
                )
                aux[f"term_{term.name}_voxel_frac"] = voxel_frac.mean().detach()

        # Sentinel diagnostics: always log healthy / wt residual_lp_mean even when
        # not in `terms`, so post-hoc analyses can compare across configs.
        with torch.no_grad():
            if m_brain is not None:
                healthy_mask = m_brain * (1.0 - m_wt)
                aux["residual_lp_mean_healthy"] = (
                    _masked_lp_per_sample(residual, healthy_mask, 2.0).mean().detach()
                )
                aux["healthy_voxel_frac"] = (
                    (
                        healthy_mask.flatten(1).sum(dim=1)
                        / m_brain.flatten(1).sum(dim=1).clamp_min(1.0)
                    )
                    .mean()
                    .detach()
                )
            aux["residual_lp_mean_wt"] = _masked_lp_per_sample(residual, m_wt, 2.0).mean().detach()

        self._per_sample = per_sample_total.detach()
        self._aux = aux
        return per_sample_total.mean()

    def per_sample(self) -> torch.Tensor | None:
        """Return the cached ``(B,)`` per-sample contrastive from the last forward.

        Used by the LightningModule's per-cohort contrastive breakdown
        (``module.py:438+``); returns ``None`` if :meth:`forward` has not been
        called yet on this instance in the current step.
        """
        return getattr(self, "_per_sample", None)

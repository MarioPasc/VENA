"""Curriculum-aware loss factory.

Stage names follow the proposal:

* ``S1``      — :math:`\\mathcal{L}_{\text{CFM}}` only.
* ``S2``      — S1 + contrastive ROI + contrastive BG.
* ``S3``      — S2 + capped :math:`L^p` background reconstruction.
* ``skipS1``  — S2 from scratch (the curriculum-necessity ablation, proposal §5.5).

The same :class:`CompositeLoss` shape is returned in every case, so the
LightningModule never branches.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AbstractFMLoss, CompositeLoss
from .cfm import CFMLoss
from .contrastive import ContrastiveTumourLoss
from .reconstruction import CappedLpReconLoss

logger = logging.getLogger(__name__)


def _get(section: dict[str, Any] | None, key: str, default: Any) -> Any:
    if section is None:
        return default
    return section.get(key, default)


def build_loss(stage: str, cfg: dict[str, Any]) -> CompositeLoss:
    """Build the composite loss for a curriculum stage.

    Parameters
    ----------
    stage : str
        One of ``"S1"``, ``"S2"``, ``"S3"``, ``"skipS1"``.
    cfg : dict
        Loss config block from the YAML, typically::

            cfm: {weight: 1.0, reduction: "mean", norm: "l2"}
            contrastive: {weight: 0.01, lambda_roi: 0.3, lambda_bg: 1.0, delta: 2.0}
            reconstruction: {weight: 0.1, p: 4, delta: 2.0}

    Returns
    -------
    CompositeLoss
    """
    stage_norm = stage.strip()
    if stage_norm not in {"S1", "S2", "S3", "skipS1"}:
        raise ValueError(f"unknown curriculum stage {stage!r}; choose from S1/S2/S3/skipS1")

    cfm_cfg = cfg.get("cfm") or {}
    contrast_cfg = cfg.get("contrastive") or {}
    recon_cfg = cfg.get("reconstruction") or {}

    terms: dict[str, AbstractFMLoss] = {
        "cfm": CFMLoss(
            reduction=_get(cfm_cfg, "reduction", "mean"),
            norm=_get(cfm_cfg, "norm", "l2"),
        )
    }
    weights: dict[str, float] = {"cfm": float(_get(cfm_cfg, "weight", 1.0))}
    requires_perturb = False

    if stage_norm in {"S2", "S3", "skipS1"}:
        terms["contrastive"] = ContrastiveTumourLoss(
            lambda_roi=float(_get(contrast_cfg, "lambda_roi", 0.3)),
            lambda_bg=float(_get(contrast_cfg, "lambda_bg", 1.0)),
            delta=float(_get(contrast_cfg, "delta", 2.0)),
            p_t=float(_get(contrast_cfg, "p_t", 1.0)),
            p_b=float(_get(contrast_cfg, "p_b", 3.0)),
        )
        weights["contrastive"] = float(_get(contrast_cfg, "weight", 0.01))
        requires_perturb = True

    if stage_norm == "S3":
        terms["reconstruction"] = CappedLpReconLoss(
            p=int(_get(recon_cfg, "p", 4)),
            delta=float(_get(recon_cfg, "delta", 2.0)),
        )
        weights["reconstruction"] = float(_get(recon_cfg, "weight", 0.1))

    logger.info(
        "build_loss(stage=%s): terms=%s requires_perturbed_pass=%s",
        stage_norm,
        list(terms.keys()),
        requires_perturb,
    )
    return CompositeLoss(
        terms=terms,
        weights=weights,
        requires_perturbed_pass=requires_perturb,
        stage=stage_norm,
    )

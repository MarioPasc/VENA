"""Curriculum-aware loss factory (v0.5 — 2026-06-19 S3 overhaul).

Stage names follow the proposal:

* ``S1``      — :math:`\\mathcal{L}_{\\text{CFM}}` only.
* ``S2``      — S1 + the region-weighted CFM-residual contrastive
  (:class:`ContrastiveTumourLoss` v0.4). Classifier-free guidance dropout is
  configured at the training-loop level via
  ``training.conditioning_dropout_p``, not through this builder.
* ``S3``      — S1 + the decoder-feature Latent Perceptual Loss (LPL,
  :class:`vena.model.fm.lpl.LplLoss`). The contrastive term is **not** part
  of S3; LPL replaces it as the image-aware learning signal (see
  ``.claude/notes/changes/decoder_perceptual_loss_s3.md``). The LPL term is
  added in :class:`FMLightningModule.training_step` as
  ``lambda_img(epoch) * lpl_scalar``, never through :class:`CompositeLoss` —
  so the builder for S3 returns a CFM-only composite. The legacy
  ``CappedLpReconLoss`` stub is no longer instantiated by any stage; the
  v0.4 proposal §5.4 "capped L^p" idea was superseded by LPL.
* ``skipS1``  — S2 from scratch (the curriculum-necessity ablation,
  proposal §5.5).

The same :class:`CompositeLoss` shape is returned in every case, so the
LightningModule never branches on stage.

The v0.4 ``contrastive`` block carries a list of region terms:

.. code-block:: yaml

    loss:
      contrastive:
        weight: 0.1                # outer scalar applied by CompositeLoss
        terms:
          - {name: healthy, region: healthy, p: 2.0, weight: 1.0}
          - {name: wt,      region: wt,      p: 1.0, weight: 0.5}

The legacy keys ``lambda_roi``, ``lambda_bg``, ``delta``, ``p_t``, ``p_b`` are
rejected with a clear migration error — they were the v0.3 mask-sensitivity
recipe that the 2026-06-09 overhaul retired.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AbstractFMLoss, CompositeLoss
from .cfm import CFMLoss
from .contrastive import ContrastiveTumourLoss, RegionTerm
from .schedule import (
    WeightSchedule,
    build_schedule,
)

logger = logging.getLogger(__name__)

# v0.3 contrastive params that no longer exist. Used to produce a clear
# migration error when an old YAML reaches this builder.
_LEGACY_CONTRASTIVE_KEYS: frozenset[str] = frozenset(
    {"lambda_roi", "lambda_bg", "delta", "p_t", "p_b"}
)


def _get(section: dict[str, Any] | None, key: str, default: Any) -> Any:
    if section is None:
        return default
    return section.get(key, default)


def _parse_region_terms(contrast_cfg: dict[str, Any]) -> list[RegionTerm]:
    """Parse the YAML ``contrastive.terms`` list into ``RegionTerm`` objects.

    Raises a clear migration error when the v0.3 keys are present, so a stale
    YAML cannot silently fall back to defaults.
    """
    legacy_keys_present = _LEGACY_CONTRASTIVE_KEYS & contrast_cfg.keys()
    if legacy_keys_present:
        raise ValueError(
            f"loss.contrastive carries v0.3 keys {sorted(legacy_keys_present)} that "
            "were removed in the 2026-06-09 overhaul. Replace with a `terms:` "
            "list of {name, region, p, weight} — see "
            "`routines/fm/train/configs/runs/picasso_s2_1000ep_*.yaml` for the "
            "new schema. Region kinds: wt | brain | healthy | background | full."
        )

    raw_terms = contrast_cfg.get("terms")
    if not raw_terms:
        # Default = single healthy/p=2 term (the doc default of CHANGE 2).
        return [RegionTerm(name="healthy", region="healthy", p=2.0, weight=1.0)]
    parsed: list[RegionTerm] = []
    for i, t in enumerate(raw_terms):
        if not isinstance(t, dict):
            raise ValueError(f"loss.contrastive.terms[{i}] must be a mapping; got {type(t)}")
        parsed.append(
            RegionTerm(
                name=str(t["name"]),
                region=str(t["region"]),  # type: ignore[arg-type]
                p=float(t["p"]),
                weight=float(t.get("weight", 1.0)),
            )
        )
    return parsed


def build_loss(stage: str, cfg: dict[str, Any]) -> CompositeLoss:
    """Build the composite loss for a curriculum stage.

    Parameters
    ----------
    stage : str
        One of ``"S1"``, ``"S2"``, ``"S3"``, ``"skipS1"``.
    cfg : dict
        Loss config block from the YAML, typically::

            cfm: {weight: 1.0, reduction: "mean", norm: "l2"}
            contrastive:                                # S2 / skipS1 only
              weight: 0.1
              terms:
                - {name: healthy, region: healthy, p: 2.0, weight: 1.0}

        For ``stage="S3"``, the ``contrastive`` block is ignored — the LPL
        coupling lives in a separate ``loss.lpl`` block consumed by
        :func:`routines.fm.train.engine._build_lpl_config`, not by this
        builder.

    Returns
    -------
    CompositeLoss
    """
    stage_norm = stage.strip()
    if stage_norm not in {"S1", "S2", "S3", "skipS1"}:
        raise ValueError(f"unknown curriculum stage {stage!r}; choose from S1/S2/S3/skipS1")

    cfm_cfg = cfg.get("cfm") or {}
    contrast_cfg = cfg.get("contrastive") or {}

    terms: dict[str, AbstractFMLoss] = {
        "cfm": CFMLoss(
            reduction=_get(cfm_cfg, "reduction", "mean"),
            norm=_get(cfm_cfg, "norm", "l2"),
        )
    }
    weights: dict[str, WeightSchedule] = {
        "cfm": build_schedule(_get(cfm_cfg, "weight", 1.0), _get(cfm_cfg, "schedule", None)),
    }
    # v0.4: the contrastive no longer needs a perturbed pass. CFG dropout (when
    # active) is handled at the training-loop level, not through the loss.
    requires_perturb = False

    # S3 carries CFM only at the CompositeLoss level. The decoder-feature LPL
    # term is added in FMLightningModule.training_step as
    # lambda_img(epoch) * lpl_scalar, gated by t > t_min, and is not part of
    # this composite — see vena.model.fm.lpl + .claude/notes/changes/decoder_perceptual_loss_s3.md.
    if stage_norm in {"S2", "skipS1"}:
        terms["contrastive"] = ContrastiveTumourLoss(terms=_parse_region_terms(contrast_cfg))
        weights["contrastive"] = build_schedule(
            _get(contrast_cfg, "weight", 0.1),
            _get(contrast_cfg, "schedule", None),
        )

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

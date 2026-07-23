"""Segmentation loss functions: DML, CE, focal-CE, Tversky, focal-Tversky.

Wang et al., *Dice Semimetric Losses for Optimizing the Dice Score*, MICCAI
2023, arXiv:2303.16296.

DML is the only Dice-family loss that is (i) symmetric in probs and target,
(ii) non-negative, and (iii) minimal (= 0) at ``probs == target``, making it
*proper* on soft labels ``target ∈ [0, 1]``.  On hard labels (``target ∈
{0,1}``), DML reduces exactly to standard soft-Dice.

Public API
----------
dice_semimetric_loss : Dice Semimetric Loss (DML) — proper on soft labels.
ce_term              : Binary cross-entropy or focal-BCE per channel.
tversky_term         : Tversky or focal-Tversky loss per channel.
SegmentationLoss     : Composite loss nn.Module driven by :class:`LossConfig`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from vena.segmentation.config import LossConfig
from vena.segmentation.exceptions import SegLossError

if TYPE_CHECKING:
    pass

__all__ = [
    "SegmentationLoss",
    "ce_term",
    "dice_semimetric_loss",
    "tversky_term",
]

logger = logging.getLogger(__name__)

# Default focal gamma used when cfg selects a focal variant.
# Pending a ``focal_gamma`` field in LossConfig (post ITER-9 config revision).
_FOCAL_GAMMA_DEFAULT: float = 2.0


def dice_semimetric_loss(
    probs: Tensor,
    target: Tensor,
    *,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> Tensor:
    """Dice Semimetric Loss (DML) for soft probability targets.

    Wang et al. (2023) define DML as:

    .. math::

        \\text{DML}(p, t) = 1 -
        \\frac{2 \\sum_i p_i t_i + \\varepsilon}
             {\\sum_i p_i^2 + \\sum_i t_i^2 + \\varepsilon}

    **Key properties (proved via Cauchy-Schwarz + AM-GM):**

    * *Symmetry*: DML(p, t) = DML(t, p).
    * *Non-negativity*: DML(p, t) ≥ 0.
    * *Properness*: DML(p, t) = 0 iff p = t (minimum uniquely at p = t).
    * *Hard-label reduction*: when both ``probs`` and ``target`` are hard
      (values in {0, 1}), ``p_i^2 = p_i`` and ``t_i^2 = t_i``, so the
      denominator collapses to ``sum(p) + sum(t)`` — identical to standard
      soft-Dice.

    MONAI's ``DiceLoss`` uses ``sum(p) + sum(t)`` in the denominator even for
    soft targets, so it is **not** minimized at ``p = t`` for soft ``t`` (the
    minimum is at the hard threshold ``p = 1``, not at ``p = t``).  DML fixes
    this by using squared norms (Euclidean inner-product form).

    Parameters
    ----------
    probs : Tensor
        Predicted probabilities ``sigmoid(logits)``, shape
        ``(B, C, *spatial)``.  Values should be in ``[0, 1]``.
    target : Tensor
        Soft targets in ``[0, 1]``, same shape as *probs*.
    reduction : str
        ``"mean"`` — mean over (B, C) pairs (default);
        ``"sum"`` — sum; ``"none"`` — return per-(B, C) tensor.
    eps : float
        Denominator epsilon to prevent division by zero.

    Returns
    -------
    Tensor
        Scalar when ``reduction ∈ {"mean", "sum"}``; shape ``(B, C)``
        when ``reduction == "none"``.

    Raises
    ------
    SegLossError
        If *probs* and *target* have different shapes, *probs* is < 2-D,
        or *reduction* is unrecognised.
    """
    if probs.shape != target.shape:
        raise SegLossError(
            f"probs and target must have the same shape; got {probs.shape} vs {target.shape}"
        )
    if probs.dim() < 2:
        raise SegLossError(f"probs must be at least 2-D (B, C, ...); got {probs.dim()}-D")
    if reduction not in {"mean", "sum", "none"}:
        raise SegLossError(f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}")

    b, c = probs.shape[0], probs.shape[1]
    p = probs.reshape(b, c, -1)  # (B, C, N)
    t = target.reshape(b, c, -1)

    numerator = 2.0 * (p * t).sum(dim=-1)  # (B, C)
    denominator = (p * p).sum(dim=-1) + (t * t).sum(dim=-1)  # (B, C)
    dml = 1.0 - (numerator + eps) / (denominator + eps)  # (B, C)

    if reduction == "none":
        return dml
    if reduction == "sum":
        return dml.sum()
    return dml.mean()


def ce_term(
    logits: Tensor,
    target: Tensor,
    *,
    focal_gamma: float | None,
) -> Tensor:
    """Binary cross-entropy (or focal-BCE) per channel, mean-reduced.

    Independent sigmoid probabilities are used for each channel (TC and NETC
    are nested, not mutually exclusive, so softmax is incorrect here).  The
    numerically stable ``F.binary_cross_entropy_with_logits`` path avoids
    explicit ``log(p)`` calls that would require clamping.

    When *focal_gamma* is not None, per-element weighting follows the
    original focal loss formulation (Lin et al. 2017):

    .. math::

        \\ell_\\text{focal}(p, t) =
        \\bigl[t (1-p)^\\gamma + (1-t) p^\\gamma\\bigr] \\cdot \\ell_\\text{BCE}(p, t)

    for soft ``t ∈ [0, 1]`` the weighting interpolates between the two
    focal cases by target value.

    Focal-CE is the **primary training-time calibration lever** after
    post-hoc temperature scaling was dropped (Q5, ITER-9).

    Parameters
    ----------
    logits : Tensor
        Raw model outputs (pre-sigmoid), shape ``(B, C, *spatial)``.
    target : Tensor
        Soft targets in ``[0, 1]``, same shape as *logits*.
    focal_gamma : float or None
        Focal modulating factor γ ≥ 0.  ``None`` → standard BCE.

    Returns
    -------
    Tensor
        Scalar — mean over all elements.

    Raises
    ------
    SegLossError
        If *logits* and *target* have different shapes.
    """
    if logits.shape != target.shape:
        raise SegLossError(
            f"logits and target must have the same shape; got {logits.shape} vs {target.shape}"
        )

    # Numerically stable BCE — no explicit sigmoid or log needed here.
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    if focal_gamma is None:
        return bce.mean()

    # Focal weight: (1-p)^gamma for positives, p^gamma for negatives.
    # For soft target, interpolate by target value so the weight is consistent
    # with the limit behaviour: at t=1 → (1-p)^gamma; at t=0 → p^gamma.
    p = torch.sigmoid(logits)
    focal_weight = target * (1.0 - p) ** focal_gamma + (1.0 - target) * p**focal_gamma
    return (focal_weight * bce).mean()


def tversky_term(
    probs: Tensor,
    target: Tensor,
    *,
    alpha: float,
    beta: float,
    focal_gamma: float | None,
) -> Tensor:
    """(Focal-)Tversky loss per channel, mean-reduced.

    Salehi et al. (2017), *Tversky Loss Function for Image Segmentation Using
    3D Fully Convolutional Deep Networks*.

    .. math::

        TI = \\frac{TP + \\varepsilon}
                   {TP + \\alpha \\cdot FP + \\beta \\cdot FN + \\varepsilon}

        \\text{TverskyLoss} = 1 - TI
        \\quad \\bigl(\\text{focal variant: } (1 - TI)^\\gamma\\bigr)

    where TP = sum(p · t), FP = sum(p · (1-t)), FN = sum((1-p) · t).

    **FN weighting**: with ``alpha < beta``, a false-negative-heavy prediction
    (predicts 0 where the true label is 1) incurs *strictly* larger loss than
    a false-positive-heavy prediction of equal total count.  This is the
    desired behaviour for recall-focused tumour-core segmentation.

    Parameters
    ----------
    probs : Tensor
        Predicted probabilities in ``[0, 1]``, shape ``(B, C, *spatial)``.
    target : Tensor
        Soft targets in ``[0, 1]``, same shape as *probs*.
    alpha : float
        False-positive weight.
    beta : float
        False-negative weight.  Set ``beta > alpha`` for recall focus.
    focal_gamma : float or None
        Focal exponent applied to ``(1 - TI)``; ``None`` disables focal.

    Returns
    -------
    Tensor
        Scalar Tversky (or focal-Tversky) loss.

    Raises
    ------
    SegLossError
        If *probs* and *target* have different shapes or *probs* is < 2-D.
    """
    if probs.shape != target.shape:
        raise SegLossError(
            f"probs and target must have the same shape; got {probs.shape} vs {target.shape}"
        )
    if probs.dim() < 2:
        raise SegLossError(f"probs must be at least 2-D (B, C, ...); got {probs.dim()}-D")

    b, c = probs.shape[0], probs.shape[1]
    p = probs.reshape(b, c, -1)  # (B, C, N)
    t = target.reshape(b, c, -1)

    eps = 1e-6
    tp = (p * t).sum(dim=-1)  # (B, C)
    fp = (p * (1.0 - t)).sum(dim=-1)  # (B, C)
    fn = ((1.0 - p) * t).sum(dim=-1)  # (B, C)

    tversky_index = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = 1.0 - tversky_index  # (B, C)

    if focal_gamma is not None:
        loss = loss**focal_gamma

    return loss.mean()


def _soft_dice_loss(
    probs: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Standard soft-Dice loss (L1 denominator, improper on soft labels).

    Used by :class:`SegmentationLoss` when ``cfg.dice_variant == "soft_dice"``.
    Exposed at module level (not nested in loops) per coding-standards rule 16.

    Parameters
    ----------
    probs : Tensor
        Predicted probabilities, shape ``(B, C, *spatial)``.
    target : Tensor
        Soft targets in ``[0, 1]``, same shape as *probs*.
    eps : float
        Denominator epsilon.

    Returns
    -------
    Tensor
        Scalar soft-Dice loss.
    """
    b, c = probs.shape[0], probs.shape[1]
    p = probs.reshape(b, c, -1)
    t = target.reshape(b, c, -1)
    intersection = (p * t).sum(dim=-1)
    denom = p.sum(dim=-1) + t.sum(dim=-1)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def _compute_single_loss(
    logits: Tensor,
    target: Tensor,
    cfg: LossConfig,
    focal_gamma: float,
) -> Tensor:
    """Composite Dice-family + CE loss for one (logits, target) pair.

    Module-level helper (not nested in a loop or class method) per
    coding-standards rule 16.  Called by :class:`SegmentationLoss` for
    the main head and each deep-supervision auxiliary head.

    Parameters
    ----------
    logits : Tensor
        Raw model logits, shape ``(B, C, *spatial)``.
    target : Tensor
        Soft targets in ``[0, 1]``, same shape as *logits*.
    cfg : LossConfig
        Frozen loss configuration.
    focal_gamma : float
        Gamma used for focal variants of CE and Tversky.

    Returns
    -------
    Tensor
        Scalar composite loss.
    """
    # Independent per-channel sigmoid — TC and NETC are nested, not exclusive.
    probs = torch.sigmoid(logits)

    # --- Dice-family term ---
    variant = cfg.dice_variant
    if variant == "dml":
        dice = dice_semimetric_loss(probs, target, reduction="mean")
    elif variant == "soft_dice":
        dice = _soft_dice_loss(probs, target)
    elif variant == "tversky":
        dice = tversky_term(
            probs,
            target,
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            focal_gamma=None,
        )
    elif variant == "focal_tversky":
        dice = tversky_term(
            probs,
            target,
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            focal_gamma=focal_gamma,
        )
    else:
        raise SegLossError(f"Unknown dice_variant: {variant!r}")

    # --- CE term ---
    gamma_ce = focal_gamma if cfg.ce_variant == "focal_ce" else None
    ce = ce_term(logits, target, focal_gamma=gamma_ce)

    return cfg.dice_weight * dice + cfg.ce_weight * ce


class SegmentationLoss(nn.Module):
    """Composite segmentation loss combining a Dice-family term and CE/focal-CE.

    Supports single-head and multi-head deep supervision.  When *outputs* is a
    tuple ``(main_logits, aux1_logits, …)``, the total loss aggregates over
    all heads with per-scale weights from ``cfg.deep_supervision_weights``:

    .. math::

        \\mathcal{L} = \\sum_{i=0}^{K-1}
            w_i \\cdot \\mathcal{L}\\bigl(\\text{head}_i,
                                           \\text{target} \\downarrow_i\\bigr)

    The soft target is downsampled to each auxiliary head's spatial resolution
    via area-averaging (``F.interpolate(mode="area")``), which preserves the
    probabilistic interpretation of soft values.

    Independent sigmoid activations — **not softmax** — are applied per channel
    because the TC and NETC channels are anatomically nested (NETC ⊆ TC), not
    mutually exclusive classes.

    Parameters
    ----------
    cfg : LossConfig
        Frozen Pydantic configuration controlling variant selection and
        per-term weights.  See :class:`~vena.segmentation.config.LossConfig`.

    Notes
    -----
    Focal gamma for ``"focal_ce"`` and ``"focal_tversky"`` variants defaults
    to :data:`_FOCAL_GAMMA_DEFAULT` = 2.0, pending a ``focal_gamma`` field
    being added to :class:`~vena.segmentation.config.LossConfig`.

    Raises
    ------
    SegLossError
        If ``cfg.dice_weight`` or ``cfg.ce_weight`` is negative, or if
        ``cfg.deep_supervision_weights`` is empty.
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        if cfg.dice_weight < 0.0 or cfg.ce_weight < 0.0:
            raise SegLossError(
                "dice_weight and ce_weight must be non-negative; "
                f"got dice_weight={cfg.dice_weight}, ce_weight={cfg.ce_weight}"
            )
        if not cfg.deep_supervision_weights:
            raise SegLossError("deep_supervision_weights must not be empty.")

        self._cfg = cfg
        # Focal gamma: hard-coded default until LossConfig grows a focal_gamma field.
        self._focal_gamma: float = _FOCAL_GAMMA_DEFAULT
        logger.debug(
            "SegmentationLoss: dice=%s ce=%s dice_w=%.2f ce_w=%.2f focal_gamma=%.2f ds_weights=%s",
            cfg.dice_variant,
            cfg.ce_variant,
            cfg.dice_weight,
            cfg.ce_weight,
            self._focal_gamma,
            cfg.deep_supervision_weights,
        )

    def forward(
        self,
        outputs: Tensor | tuple[Tensor, ...],
        target: Tensor,
    ) -> Tensor:
        """Compute composite loss, aggregating deep-supervision heads if present.

        Parameters
        ----------
        outputs : Tensor or tuple of Tensor
            Single logit tensor ``(B, C, *spatial)`` **or** a tuple
            ``(main_logits, aux1_logits, …)`` for deep supervision.
            Auxiliary heads may have smaller spatial dimensions; they are
            matched to *target* via area-averaged downsampling.
        target : Tensor
            Soft targets ``(B, C, *spatial)`` in ``[0, 1]``.

        Returns
        -------
        Tensor
            Scalar composite loss with valid autograd graph.

        Raises
        ------
        SegLossError
            If the number of output heads exceeds
            ``len(cfg.deep_supervision_weights)``.
        """
        cfg = self._cfg
        gamma = self._focal_gamma

        if isinstance(outputs, Tensor):
            return _compute_single_loss(outputs, target, cfg, gamma)

        # Deep supervision — iterate heads with per-scale weights.
        heads = tuple(outputs)
        weights = cfg.deep_supervision_weights
        if len(heads) > len(weights):
            raise SegLossError(
                f"Number of output heads ({len(heads)}) exceeds "
                f"deep_supervision_weights length ({len(weights)}). "
                "Extend cfg.deep_supervision_weights."
            )

        total: Tensor = target.new_zeros(())
        for logits_i, w_i in zip(heads, weights, strict=False):
            spatial_i = logits_i.shape[2:]
            spatial_t = target.shape[2:]
            if spatial_i != spatial_t:
                # Area-avg downsampling preserves soft probabilities.
                t_i = F.interpolate(
                    target.float(),
                    size=spatial_i,
                    mode="area",
                )
            else:
                t_i = target
            total = total + w_i * _compute_single_loss(logits_i, t_i, cfg, gamma)

        return total

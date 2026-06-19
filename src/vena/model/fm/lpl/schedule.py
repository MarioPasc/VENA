"""Lambda_img schedule for the LPL coupling weight.

The S3 trainer combines CFM and LPL as

.. math::
    \\mathcal L_{\\text{S3}} = \\mathcal L_{\\text{CFM}} + \\lambda_{\\text{img}}(e) \\cdot \\mathcal L_{\\text{dec}}

where :math:`\\lambda_{\\text{img}}(e)` is an epoch-driven schedule. This
module defines the schedule shape and the pure :func:`compute_lambda_img`
evaluator. Production runs use ``kind="linear"`` with a 30-epoch warm-up
from 0 to 1; the other shapes (sigmoid, cosine-with-anneal) are wired so
ablations can swap them in without code changes.

Design rationale (per ``.claude/notes/changes/decoder_perceptual_loss_s3.md``
§5.3 and the 2026-06-19 plan): the warm-start from the converged S1 checkpoint
is robust to a soft λ landing — λ ramps up over a small fraction of total
training (~3% on a 1000-ep run), then holds at ``lambda_max``. We accept
this is more aggressive than Berrada-2025 (constant ≈3.0 on natural images,
which translates to ~1.0 on our cropped-volume 3D regime); the CFM term
remains active throughout as the latent anchor.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

LambdaImgScheduleKind = Literal["constant", "linear", "sigmoid", "cosine_with_anneal"]


class LambdaImgSchedule(BaseModel):
    """Schedule for the LPL outer coupling weight :math:`\\lambda_{\\text{img}}(e)`.

    Fields
    ------
    kind : str
        ``"constant"``, ``"linear"``, ``"sigmoid"``, or ``"cosine_with_anneal"``.
    warmup_epochs : int
        Number of epochs over which :math:`\\lambda` ramps from ``lambda_min``
        to ``lambda_max``. Ignored when ``kind="constant"``. For
        ``cosine_with_anneal`` this is the warm-up phase only — the anneal
        phase spans ``total_epochs - warmup_epochs``.
    lambda_min : float
        Initial value at epoch 0 (or asymptote at e→-∞ for ``sigmoid``).
    lambda_max : float
        Steady-state value reached at epoch ``warmup_epochs``. For
        ``cosine_with_anneal``, the peak before the anneal phase begins.
    total_epochs : int | None
        Required for ``kind="cosine_with_anneal"`` (used as the anneal end).
        Ignored otherwise.
    sigmoid_slope_k : float
        Sigmoid steepness. Higher = sharper transition. Default 0.3 gives a
        smooth ramp centred at ``warmup_epochs/2`` and reaching ~99% of the
        max by epoch ``warmup_epochs``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: LambdaImgScheduleKind = "constant"
    warmup_epochs: int = 0
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    total_epochs: int | None = None
    sigmoid_slope_k: float = 0.3

    @model_validator(mode="after")
    def _check(self) -> LambdaImgSchedule:
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0; got {self.warmup_epochs}")
        if self.lambda_min < 0.0 or self.lambda_max < 0.0:
            raise ValueError("lambda_min and lambda_max must be >= 0")
        if self.lambda_max < self.lambda_min:
            raise ValueError(
                f"lambda_max ({self.lambda_max}) must be >= lambda_min ({self.lambda_min})"
            )
        if self.kind == "cosine_with_anneal":
            if self.total_epochs is None:
                raise ValueError("kind='cosine_with_anneal' requires total_epochs (anneal end)")
            if self.total_epochs <= self.warmup_epochs:
                raise ValueError(
                    f"total_epochs ({self.total_epochs}) must exceed "
                    f"warmup_epochs ({self.warmup_epochs}) for cosine_with_anneal"
                )
        if self.sigmoid_slope_k <= 0.0:
            raise ValueError(f"sigmoid_slope_k must be > 0; got {self.sigmoid_slope_k}")
        return self


def compute_lambda_img(schedule: LambdaImgSchedule, current_epoch: int) -> float:
    """Evaluate the schedule at ``current_epoch`` (0-indexed).

    Parameters
    ----------
    schedule : LambdaImgSchedule
        The validated schedule spec.
    current_epoch : int
        Lightning's ``trainer.current_epoch`` — 0 at the very first batch of
        the very first epoch.

    Returns
    -------
    float
        Non-negative scalar; always within ``[lambda_min, lambda_max]`` for
        every schedule kind except ``cosine_with_anneal``, which can decay
        below ``lambda_max`` toward ``lambda_min`` after the warm-up.
    """
    e = max(0, int(current_epoch))
    if schedule.kind == "constant":
        return float(schedule.lambda_max)
    if schedule.kind == "linear":
        if schedule.warmup_epochs == 0:
            return float(schedule.lambda_max)
        if e >= schedule.warmup_epochs:
            return float(schedule.lambda_max)
        frac = float(e) / float(schedule.warmup_epochs)
        return float(schedule.lambda_min + frac * (schedule.lambda_max - schedule.lambda_min))
    if schedule.kind == "sigmoid":
        midpoint = float(schedule.warmup_epochs) / 2.0 if schedule.warmup_epochs > 0 else 0.0
        x = schedule.sigmoid_slope_k * (float(e) - midpoint)
        sig = 1.0 / (1.0 + math.exp(-x))
        return float(schedule.lambda_min + sig * (schedule.lambda_max - schedule.lambda_min))
    if schedule.kind == "cosine_with_anneal":
        # warm-up phase: cosine 0 → 1 over [0, warmup_epochs]
        if e <= schedule.warmup_epochs:
            if schedule.warmup_epochs == 0:
                return float(schedule.lambda_max)
            frac = float(e) / float(schedule.warmup_epochs)
            # cosine warm-up (0 → 1 as cosine flips from -1 → 1)
            cos_in = 0.5 * (1.0 - math.cos(math.pi * frac))
            return float(schedule.lambda_min + cos_in * (schedule.lambda_max - schedule.lambda_min))
        # anneal phase: cosine lambda_max → lambda_min over [warmup, total]
        assert schedule.total_epochs is not None
        span = float(schedule.total_epochs - schedule.warmup_epochs)
        post = float(min(e, schedule.total_epochs) - schedule.warmup_epochs)
        cos_out = 0.5 * (1.0 + math.cos(math.pi * post / span))  # 1 → 0
        return float(schedule.lambda_min + cos_out * (schedule.lambda_max - schedule.lambda_min))
    raise ValueError(f"unknown LambdaImgSchedule.kind={schedule.kind!r}")


__all__ = ["LambdaImgSchedule", "LambdaImgScheduleKind", "compute_lambda_img"]

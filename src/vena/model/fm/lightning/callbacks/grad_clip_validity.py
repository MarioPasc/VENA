"""§18 early-abort validity callback: grad_clip_active mean < 5 %.

Checking after ``trainer.fit()`` returns catches an invalid arm too late — the
full training budget (~5 A100-days) would already be spent.  This callback
checks once when ``global_step`` reaches ``check_step`` (default 10 000, ~1 %
of the 800 000-step budget) and raises immediately so SLURM can report a
clean failure within a few hours rather than at the end of the run.

The post-training ``_assert_grad_clip_validity`` call in the engine is kept as
belt-and-braces (e.g. for debug runs that finish before ``check_step``).

Metric key: ``train/grad_clip_active`` — a float in {0.0, 1.0} logged once per
optimiser step by ``FMLightningModule.configure_gradient_clipping``.  Values
are accumulated from steps strictly past ``warmup_step`` (default 5 000), so
the early-training spike in clip frequency does not inflate the estimate.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

_METRIC_KEY = "train/grad_clip_active"


class GradClipValidityCallback(pl.Callback):
    """§18 validity criterion — abort early if the arm clips too frequently.

    Parameters
    ----------
    tag : str
        The run tag (``cfg.run.tag``).  Included in the AssertionError message
        so the failure is actionable from the SLURM log alone.
    check_step : int
        Global step at which to evaluate the criterion (default 10 000).
    threshold : float
        Maximum allowed mean of ``train/grad_clip_active`` (default 0.05).
    warmup_step : int
        Steps before this value are excluded from the mean (default 5 000).
    """

    def __init__(
        self,
        tag: str,
        *,
        check_step: int = 10_000,
        threshold: float = 0.05,
        warmup_step: int = 5_000,
    ) -> None:
        super().__init__()
        self.tag = tag
        self.check_step = check_step
        self.threshold = threshold
        self.warmup_step = warmup_step
        # Accumulated grad_clip_active values strictly past warmup_step.
        self._history: list[float] = []
        # Last global_step for which we wrote a row (grad-accum gate).
        self._last_step: int = 0
        # True once we have evaluated (prevents a second fire on the same run).
        self._checked: bool = False

    # ------------------------------------------------------------------
    # Per micro-batch — gated to fire only on real optimiser steps
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: object,
    ) -> None:
        if self._checked:
            return
        step = int(trainer.global_step)
        # Same gradient-accumulation gate as TrainMetricsCSV: only one row per
        # optimiser step regardless of how many micro-batches compose it.
        if step <= self._last_step:
            return
        self._last_step = step

        # Accumulate only past the warm-up window.
        if step > self.warmup_step:
            val = trainer.callback_metrics.get(_METRIC_KEY)
            if val is not None:
                self._history.append(float(val))

        if step < self.check_step or not self._history:
            return

        mean_clip = sum(self._history) / len(self._history)
        self._checked = True

        if mean_clip >= self.threshold:
            raise AssertionError(
                f"[§18 grad_clip_active validity] arm={self.tag!r}: "
                f"mean grad_clip_active = {mean_clip:.4f} >= {self.threshold:.2f} "
                f"over {len(self._history)} steps "
                f"(warmup_step={self.warmup_step}, checked at step={step}). "
                "The arm clips too frequently — raise gradient_clip_val or reduce LR. "
                "Do NOT report this arm without investigation."
            )
        logger.info(
            "[§18 grad_clip_active validity] arm=%r PASS — "
            "mean=%.4f < %.2f over %d steps (checked at step=%d)",
            self.tag,
            mean_clip,
            self.threshold,
            len(self._history),
            step,
        )

"""Sigmoid scale ramp on the ControlNet output projections.

Modulates :attr:`MaisiControlNet.output_scale` (a non-persistent buffer
multiplied into every down-block residual and the mid-block residual at
:meth:`MaisiControlNet.forward`) from ~0 to ~1 over the first ``ramp_steps``
optimisation steps. Removes the cold-start dead time the literal-zero-init
ControlNet otherwise pays — at step 0 the residual contribution is ~0.7%
of the natural magnitude (a value small enough that the pretrained MAISI
trunk is undisturbed), growing to 99.3% at step ``ramp_steps``.

Reference: 2026-06-20 analysis §4a (E1) at
``.claude/notes/changes/decoder_perceptual_loss_s3_analysis_2026-06-20.md``.
The cold-start argument vs T1C-RFlow (channel-concat conditioning that is
active from step 1) is in §4 of the same document.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl

if TYPE_CHECKING:
    import torch  # noqa: F401

logger = logging.getLogger(__name__)


class OutputScaleRampCallback(pl.Callback):
    """Sigmoid ramp of ``MaisiControlNet.output_scale`` from ~0 to 1.

    Parameters
    ----------
    ramp_steps : int
        Number of training steps over which the ramp completes. After this
        many ``global_step`` increments the buffer is clamped to ``1.0`` and
        the callback becomes a no-op. Default ``5000``.
    steepness : float
        Sigmoid steepness ``k`` in ``sigmoid(k * (step / ramp_steps - 0.5))``.
        Higher ``k`` makes the transition sharper around the midpoint.
        Default ``10.0`` (value at step 0 ≈ 0.0067; at midpoint = 0.5; at
        ``ramp_steps`` ≈ 0.9933).

    Notes
    -----
    The buffer is non-persistent (not saved to checkpoints): the ramp value
    is a pure function of ``trainer.global_step`` and is recomputed on
    resume. ``global_step`` is restored by Lightning from ``last.ckpt``, so
    a resumed run picks the ramp up at the correct value on the first batch.
    """

    def __init__(self, ramp_steps: int = 5000, steepness: float = 10.0) -> None:
        super().__init__()
        if ramp_steps <= 0:
            raise ValueError(f"ramp_steps must be positive; got {ramp_steps}")
        self.ramp_steps = int(ramp_steps)
        self.steepness = float(steepness)
        self._logged_value: float | None = None

    def ramp_value(self, step: int) -> float:
        """Return the scalar to multiply into the ControlNet outputs at ``step``.

        Clamped to ``1.0`` for ``step >= ramp_steps`` so the callback turns
        into an idempotent no-op (the buffer stays at 1.0).
        """
        if step >= self.ramp_steps:
            return 1.0
        progress = float(step) / float(self.ramp_steps)
        return 1.0 / (1.0 + math.exp(-self.steepness * (progress - 0.5)))

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        controlnet = getattr(pl_module, "controlnet", None)
        if controlnet is None:
            return
        scale_buffer = getattr(controlnet, "output_scale", None)
        if scale_buffer is None:
            return
        step = int(trainer.global_step)
        value = self.ramp_value(step)
        scale_buffer.fill_(value)
        # One-line milestone logs: first batch and ramp completion. Keeps the
        # train.log narrative readable without polluting per-step logging.
        if self._logged_value is None:
            logger.info(
                "OutputScaleRampCallback: step=%d output_scale=%.4f (ramp_steps=%d steepness=%.1f)",
                step,
                value,
                self.ramp_steps,
                self.steepness,
            )
            self._logged_value = value
        elif step == self.ramp_steps and self._logged_value < 1.0:
            logger.info(
                "OutputScaleRampCallback: ramp complete at step=%d (output_scale=1.0)",
                step,
            )
            self._logged_value = 1.0

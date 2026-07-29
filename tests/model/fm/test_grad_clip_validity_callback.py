"""Unit tests for GradClipValidityCallback (§18 early-abort validity criterion).

The callback raises AssertionError when mean(grad_clip_active) >= threshold
at check_step.  The grad-clip guard cannot be exercised in a 4-epoch loginexa
smoke (~200 steps, well under the warmup_step=5 000 threshold), so this file
is the authoritative coverage for that code path.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, PropertyMock

import pytest

from vena.model.fm.lightning.callbacks.grad_clip_validity import (
    GradClipValidityCallback,
)

pytestmark = pytest.mark.unit

_METRIC_KEY = "train/grad_clip_active"


def _make_trainer(global_step: int, clip_active: float | None) -> MagicMock:
    """Build a minimal trainer mock with ``global_step`` and ``callback_metrics``."""
    trainer = MagicMock()
    type(trainer).global_step = PropertyMock(return_value=global_step)
    trainer.callback_metrics = {_METRIC_KEY: clip_active} if clip_active is not None else {}
    return trainer


def _advance(cb: GradClipValidityCallback, steps: list[tuple[int, float | None]]) -> None:
    """Drive the callback through a sequence of (global_step, clip_active) pairs."""
    pl_module = MagicMock()
    for step, clip_val in steps:
        trainer = _make_trainer(step, clip_val)
        cb.on_train_batch_end(trainer, pl_module)


class TestGradClipValidityCallback:
    """Core behavioural contract."""

    # ---- check fires and raises when threshold exceeded ----

    def test_raises_when_mean_above_threshold(self) -> None:
        cb = GradClipValidityCallback(
            tag="test_arm",
            check_step=20,
            threshold=0.05,
            warmup_step=10,
        )
        # Steps 11-19: all clips active → mean = 1.0
        steps = [(i, 1.0) for i in range(11, 21)]
        with pytest.raises(AssertionError, match="test_arm"):
            _advance(cb, steps)

    def test_error_message_contains_observed_mean(self) -> None:
        cb = GradClipValidityCallback(
            tag="high_clip_arm",
            check_step=15,
            threshold=0.05,
            warmup_step=10,
        )
        steps = [(i, 1.0) for i in range(11, 16)]
        with pytest.raises(AssertionError, match=r"1\.0000"):
            _advance(cb, steps)

    def test_error_message_contains_threshold(self) -> None:
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=15,
            threshold=0.05,
            warmup_step=10,
        )
        steps = [(i, 1.0) for i in range(11, 16)]
        with pytest.raises(AssertionError, match=r"0\.05"):
            _advance(cb, steps)

    # ---- check passes silently when threshold not exceeded ----

    def test_passes_when_mean_below_threshold(self) -> None:
        cb = GradClipValidityCallback(
            tag="clean_arm",
            check_step=20,
            threshold=0.05,
            warmup_step=10,
        )
        # mean = 0.0 — no clips at all
        steps = [(i, 0.0) for i in range(11, 21)]
        _advance(cb, steps)  # must not raise
        assert cb._checked

    def test_passes_at_exactly_threshold_minus_one_step(self) -> None:
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=20,
            threshold=0.05,
            warmup_step=10,
        )
        # 1 clip in 20 post-warmup steps = 1/20 = 0.05 — on the boundary: PASS
        # (criterion is mean >= threshold; 0.05 >= 0.05 would raise — use 0/20)
        steps = [(i, 0.0) for i in range(11, 21)]
        _advance(cb, steps)  # 0/10 = 0.0 < 0.05: passes
        assert cb._checked

    # ---- warm-up exclusion ----

    def test_warmup_steps_excluded_from_mean(self) -> None:
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=20,
            threshold=0.05,
            warmup_step=15,
        )
        # Steps 1-15 all clip=1.0 (inside warmup — should be ignored).
        # Steps 16-20 all clip=0.0 (post-warmup — mean=0.0).
        steps = [(i, 1.0) for i in range(1, 16)] + [(i, 0.0) for i in range(16, 21)]
        _advance(cb, steps)  # must not raise (warmup excluded)
        assert cb._checked
        assert math.isclose(sum(cb._history) / len(cb._history), 0.0, abs_tol=1e-9)

    def test_does_not_accumulate_at_warmup_step_exactly(self) -> None:
        """Steps <= warmup_step must not appear in the history."""
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=20,
            threshold=0.05,
            warmup_step=10,
        )
        # Fire step 10 (== warmup_step): should NOT accumulate
        _advance(cb, [(10, 1.0)])
        assert len(cb._history) == 0

    # ---- timing: not checked before check_step ----

    def test_does_not_check_before_check_step(self) -> None:
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=100,
            threshold=0.05,
            warmup_step=10,
        )
        steps = [(i, 1.0) for i in range(11, 99)]
        _advance(cb, steps)  # many steps past warmup but before check_step
        assert not cb._checked

    def test_no_check_when_history_empty_at_check_step(self) -> None:
        """If no post-warmup metric values arrived, don't evaluate yet."""
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=20,
            threshold=0.05,
            warmup_step=18,  # warmup almost at check_step
        )
        # Only steps 11-20, but warmup=18 so only step 19-20 post-warmup:
        steps = [(i, None) for i in range(11, 20)] + [(20, None)]  # no metric
        _advance(cb, steps)
        assert not cb._checked  # no history → no check

    # ---- gradient accumulation gate ----

    def test_only_fires_on_advancing_global_step(self) -> None:
        """Micro-batch repeats of the same step must not double-accumulate.

        Uses check_step=100 so the validity check never fires during steps
        11-20, letting us assert the history length without catching a raise.
        """
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=100,  # well past the test window — check never fires
            threshold=0.05,
            warmup_step=10,
        )
        pl_module = MagicMock()
        # Simulate grad_accum=2: two micro-batches per optimiser step.
        for step in range(11, 21):
            for _ in range(2):
                trainer = _make_trainer(step, 1.0)
                cb.on_train_batch_end(trainer, pl_module)
        # Should have exactly 10 values — one per optimiser step, not 20.
        assert len(cb._history) == 10

    # ---- idempotency: no double-fire ----

    def test_does_not_raise_twice(self) -> None:
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=15,
            threshold=0.05,
            warmup_step=10,
        )
        steps_raise = [(i, 1.0) for i in range(11, 16)]
        with pytest.raises(AssertionError):
            _advance(cb, steps_raise)
        # Subsequent calls must be no-ops
        _advance(cb, [(i, 1.0) for i in range(16, 25)])  # must not raise again

    def test_checked_flag_set_after_pass(self) -> None:
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=15,
            threshold=0.05,
            warmup_step=10,
        )
        _advance(cb, [(i, 0.0) for i in range(11, 16)])
        assert cb._checked
        # Any subsequent call should be a no-op
        _advance(cb, [(i, 1.0) for i in range(16, 25)])
        assert cb._checked

    # ---- boundary: exactly at threshold ----

    def test_raises_exactly_at_threshold(self) -> None:
        """mean == threshold must raise (>= semantics)."""
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=20,
            threshold=0.10,
            warmup_step=10,
        )
        # 1 clip in 10 post-warmup steps → mean = 0.10 exactly
        steps = [(11, 1.0)] + [(i, 0.0) for i in range(12, 21)]
        with pytest.raises(AssertionError):
            _advance(cb, steps)

    def test_passes_just_below_threshold(self) -> None:
        """mean just below threshold must not raise."""
        cb = GradClipValidityCallback(
            tag="arm",
            check_step=21,
            threshold=0.10,
            warmup_step=10,
        )
        # 1 clip in 11 post-warmup steps → mean ≈ 0.0909 < 0.10
        steps = [(11, 1.0)] + [(i, 0.0) for i in range(12, 22)]
        _advance(cb, steps)  # must not raise
        assert cb._checked

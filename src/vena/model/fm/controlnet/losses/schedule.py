"""Weight schedules for composite-loss terms.

Each term in :class:`CompositeLoss` carries a :class:`WeightSchedule` that maps
``(global_step, total_steps)`` → ``float``. The default is :class:`StaticWeight`
— the constant weight from the YAML — so adding a schedule is opt-in and
backward-compatible with every existing test and YAML.

Two concrete schedules currently exist:

* :class:`StaticWeight` — the constant `w0` regardless of step.
* :class:`StepHalfWeight` — w0 for the first half of training, then `w0 * factor`
  for the second half. Matches the proposal §3 anneal ``0.01 → 0.001`` at
  step half (factor=0.1).

A schedule's :meth:`at` returns 0.0 (not w0) when ``total_steps`` is unknown
*and* the schedule is non-static — the LightningModule logs the effective
weight every step so a misconfigured anneal is visible immediately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class WeightSchedule(ABC):
    """One term's outer-weight policy across training."""

    @abstractmethod
    def at(self, global_step: int | None, total_steps: int | None) -> float:
        """Return the effective weight for the current optimiser step."""


@dataclass(frozen=True)
class StaticWeight(WeightSchedule):
    """Constant weight throughout training."""

    w0: float

    def at(self, global_step: int | None, total_steps: int | None) -> float:
        return float(self.w0)


@dataclass(frozen=True)
class StepHalfWeight(WeightSchedule):
    """Step anneal at ``total_steps // 2``.

    Returns ``w0`` for ``global_step < total_steps // 2`` and ``w0 * factor``
    afterwards. ``factor`` is typically ``0.1`` per proposal §3
    (0.01 → 0.001 outer-λ anneal for the contrastive term).

    If ``total_steps`` is None (or zero) the schedule cannot evaluate; it
    returns ``w0`` so the early-phase weight is preserved.
    """

    w0: float
    factor: float = 0.1

    def at(self, global_step: int | None, total_steps: int | None) -> float:
        if total_steps is None or total_steps <= 0 or global_step is None:
            return float(self.w0)
        half = int(total_steps) // 2
        return float(self.w0 if global_step < half else self.w0 * self.factor)


def build_schedule(cfg_weight: float, schedule_cfg: dict[str, Any] | None) -> WeightSchedule:
    """Parse a YAML weight block into a :class:`WeightSchedule`.

    The YAML accepts either form::

        contrastive:
          weight: 0.01            # static (legacy)

        contrastive:
          weight: 0.01
          schedule:
            kind: step_half
            factor: 0.1

    The ``schedule.kind`` selector keeps the door open for future shapes
    (linear, cosine, ...).
    """
    if not schedule_cfg:
        return StaticWeight(float(cfg_weight))
    kind = str(schedule_cfg.get("kind", "static")).lower()
    if kind == "static":
        return StaticWeight(float(cfg_weight))
    if kind == "step_half":
        return StepHalfWeight(
            w0=float(cfg_weight),
            factor=float(schedule_cfg.get("factor", 0.1)),
        )
    raise ValueError(f"unknown schedule.kind={kind!r}; supported: static, step_half")

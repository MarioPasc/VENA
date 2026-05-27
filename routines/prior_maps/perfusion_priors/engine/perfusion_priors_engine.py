"""Thin wrapper around :class:`vena.prior_maps.perfusion_priors.PerfusionPriorsEngine`.

The routine engine intentionally adds no behaviour. It exists so the
``routines/`` directory follows the layout fixed by
``.claude/rules/preflight-pattern.md``.
"""

from __future__ import annotations

from pathlib import Path

from vena.prior_maps.perfusion_priors import (
    PerfusionPriorsEngine,
    PerfusionPriorsRoutineConfig,
)


class PerfusionPriorsRoutineEngine:
    def __init__(self, cfg: PerfusionPriorsRoutineConfig) -> None:
        self._inner = PerfusionPriorsEngine(cfg)

    def run(self, *, figures_only: bool = False) -> Path:
        return self._inner.run(figures_only=figures_only)

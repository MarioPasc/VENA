"""Thin wrapper around :class:`vena.prior_maps.susceptibility_priors.SusceptibilityPriorsEngine`."""

from __future__ import annotations

from pathlib import Path

from vena.prior_maps.susceptibility_priors import (
    SusceptibilityPriorsEngine,
    SusceptibilityPriorsRoutineConfig,
)


class SusceptibilityPriorsRoutineEngine:
    def __init__(self, cfg: SusceptibilityPriorsRoutineConfig) -> None:
        self._inner = SusceptibilityPriorsEngine(cfg)

    def run(self, *, figures_only: bool = False) -> Path:
        return self._inner.run(figures_only=figures_only)

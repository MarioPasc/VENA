"""Thin wrapper around :class:`vena.prior_maps.cellularity_priors.CellularityPriorsEngine`."""

from __future__ import annotations

from pathlib import Path

from vena.prior_maps.cellularity_priors import (
    CellularityPriorsEngine,
    CellularityPriorsRoutineConfig,
)


class CellularityPriorsRoutineEngine:
    def __init__(self, cfg: CellularityPriorsRoutineConfig) -> None:
        self._inner = CellularityPriorsEngine(cfg)

    def run(self, *, figures_only: bool = False) -> Path:
        return self._inner.run(figures_only=figures_only)

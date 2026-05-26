"""Thin wrapper around :class:`vena.preflight.priors_validation.PriorsValidationEngine`."""

from __future__ import annotations

from pathlib import Path

from vena.preflight.priors_validation import (
    PriorsValidationEngine,
    PriorsValidationRoutineConfig,
)


class PriorsValidationRoutineEngine:
    def __init__(self, cfg: PriorsValidationRoutineConfig) -> None:
        self._inner = PriorsValidationEngine(cfg)

    def run(self) -> Path:
        return self._inner.run()

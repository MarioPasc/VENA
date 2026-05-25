"""Thin wrapper around :class:`vena.vessel_priors.VesselPriorsEngine`.

The routine engine intentionally adds no behaviour. It exists so the
``routines/`` directory follows the layout fixed by
``.claude/rules/preflight-pattern.md`` (one engine module per routine) while the
actual orchestration code lives under ``src/vena/`` and stays importable and
unit-testable from outside the CLI.
"""

from __future__ import annotations

from pathlib import Path

from vena.vessel_priors import VesselPriorsEngine, VesselPriorsRoutineConfig


class VesselPriorsRoutineEngine:
    def __init__(self, cfg: VesselPriorsRoutineConfig) -> None:
        self._inner = VesselPriorsEngine(cfg)

    def run(self, *, figures_only: bool = False) -> Path:
        return self._inner.run(figures_only=figures_only)

"""Thin wrapper around :class:`vena.preflight.vessel_mask.VesselMaskPreflightEngine`.

The routine engine intentionally adds no behaviour. It exists so the
``routines/`` directory follows ``.claude/rules/preflight-pattern.md`` (one
engine module per routine) while the orchestration code stays under
``src/vena/`` and remains importable / unit-testable outside the CLI.
"""

from __future__ import annotations

from pathlib import Path

from vena.preflight.vessel_mask import (
    VesselMaskPreflightConfig,
    VesselMaskPreflightEngine,
)


class VesselMaskPreflightRoutineEngine:
    def __init__(self, cfg: VesselMaskPreflightConfig) -> None:
        self._inner = VesselMaskPreflightEngine(cfg)

    def run(self) -> Path:
        return self._inner.run()

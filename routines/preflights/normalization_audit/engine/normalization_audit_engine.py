"""Thin routine wrapper around the library engine."""

from __future__ import annotations

from pathlib import Path

from vena.preflight.normalization_audit import (
    NormalizationAuditConfig,
    NormalizationAuditEngine,
)


class NormalizationAuditRoutineEngine:
    """Pass-through wrapper matching the project's routine convention."""

    def __init__(self, cfg: NormalizationAuditConfig) -> None:
        self._inner = NormalizationAuditEngine(cfg=cfg)

    def run(self) -> Path:
        return self._inner.run()

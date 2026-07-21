"""Thin routine engine for the ρ_S normalisation audit.

All heavy logic lives in ``vena.preflight.rho_s_norm_audit.engine``.
This module wires the library engine to the YAML config following the
preflight-pattern.md contract:

  - Frozen Pydantic config with ``from_yaml``
  - ``Engine.run() -> Path``
  - No heavy work at import time
"""

from __future__ import annotations

from pathlib import Path

from vena.preflight.rho_s_norm_audit import RhoSNormAuditConfig, RhoSNormAuditEngine

# Re-export the library config under the routine name so callers can import
# from either location.
RhoSNormAuditRoutineConfig = RhoSNormAuditConfig


class RhoSNormAuditRoutineEngine:
    """Thin wrapper; delegates entirely to the library engine.

    Parameters
    ----------
    cfg :
        Frozen config loaded via ``RhoSNormAuditRoutineConfig.from_yaml``.
    """

    def __init__(self, cfg: RhoSNormAuditConfig) -> None:
        self._library_engine = RhoSNormAuditEngine(cfg)

    def run(self) -> Path:
        """Run the audit and return the artifact directory path."""
        return self._library_engine.run()

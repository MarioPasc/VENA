"""Thin wrapper around :func:`vena.preflight.priors_validation.atlases.venous_build.build_venous_atlas`."""

from __future__ import annotations

from typing import Any

from vena.preflight.priors_validation.atlases.venous_build import (
    VenousBuildConfig,
    build_venous_atlas,
)


class VenousAtlasBuildRoutineEngine:
    def __init__(self, cfg: VenousBuildConfig) -> None:
        self.cfg = cfg

    def run(self) -> dict[str, Any]:
        return build_venous_atlas(self.cfg)

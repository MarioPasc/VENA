"""Thin routine wrapper around the library engine."""

from __future__ import annotations

from pathlib import Path

from vena.preflight.latent_aug_equivariance import (
    LatentAugEquivarianceConfig,
    LatentAugEquivarianceEngine,
)


class LatentAugEquivariancePreflightRoutineEngine:
    """Pass-through wrapper matching the project's routine convention."""

    def __init__(
        self,
        cfg: LatentAugEquivarianceConfig,
        config_yaml_path: Path | None = None,
    ) -> None:
        self._inner = LatentAugEquivarianceEngine(cfg=cfg, config_yaml_path=config_yaml_path)

    def run(self) -> Path:
        return self._inner.run()

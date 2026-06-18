"""Thin routine wrapper around the library engine.

Mirrors :mod:`routines.preflights.latent_aug_equivariance.engine.latent_aug_equivariance_engine`
so the routine layout is consistent across preflights.
"""

from __future__ import annotations

from pathlib import Path

from vena.preflight.decoder_lpl_profile import (
    DecoderLplProfileConfig,
    DecoderLplProfileEngine,
)


class DecoderLplProfileRoutineEngine:
    """Pass-through wrapper matching the project's routine convention."""

    def __init__(
        self,
        cfg: DecoderLplProfileConfig,
        config_yaml_path: Path | None = None,
    ) -> None:
        self._inner = DecoderLplProfileEngine(cfg=cfg, config_yaml_path=config_yaml_path)

    def run(self) -> Path:
        return self._inner.run()

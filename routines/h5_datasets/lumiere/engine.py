"""Thin routine wrapper around the LUMIERE image-domain H5 converter."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.data.h5.lumiere.image_domain import (
    LUMIEREImageH5Config,
    LUMIEREImageH5Converter,
)

logger = logging.getLogger(__name__)


class LUMIEREH5RoutineConfig(LUMIEREImageH5Config):
    """Routine-level config; identical to the library config."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> LUMIEREH5RoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


class LUMIEREH5RoutineEngine:
    def __init__(self, cfg: LUMIEREH5RoutineConfig) -> None:
        self.cfg = cfg
        self._inner = LUMIEREImageH5Converter(cfg)

    def run(self) -> Path:
        return self._inner.run()

"""Thin routine wrapper around the REMBRANDT image-domain converter."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.data.h5.rembrandt.image_domain import (
    REMBRANDTImageH5Config,
    REMBRANDTImageH5Converter,
)

logger = logging.getLogger(__name__)


class REMBRANDTH5RoutineConfig(REMBRANDTImageH5Config):
    """Routine-level config; same fields as :class:`REMBRANDTImageH5Config`."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> REMBRANDTH5RoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


class REMBRANDTH5RoutineEngine:
    def __init__(self, cfg: REMBRANDTH5RoutineConfig) -> None:
        self.cfg = cfg
        self._inner = REMBRANDTImageH5Converter(cfg)

    def run(self) -> Path:
        return self._inner.run()

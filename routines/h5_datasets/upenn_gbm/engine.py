"""Thin routine wrapper around the UPENN-GBM image-domain converter."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.data.h5.upenn_gbm.image_domain import (
    UPENNGBMImageH5Config,
    UPENNGBMImageH5Converter,
)

logger = logging.getLogger(__name__)


class UPENNGBMH5RoutineConfig(UPENNGBMImageH5Config):
    """Routine-level config; same fields as :class:`UPENNGBMImageH5Config`."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> UPENNGBMH5RoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


class UPENNGBMH5RoutineEngine:
    def __init__(self, cfg: UPENNGBMH5RoutineConfig) -> None:
        self.cfg = cfg
        self._inner = UPENNGBMImageH5Converter(cfg)

    def run(self) -> Path:
        return self._inner.run()

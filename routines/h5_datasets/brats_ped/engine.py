"""Thin routine wrapper around the BraTS-PED image-domain converter."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.data.h5.brats_ped.image_domain import (
    BraTSPedImageH5Config,
    BraTSPedImageH5Converter,
)

logger = logging.getLogger(__name__)


class BraTSPedH5RoutineConfig(BraTSPedImageH5Config):
    """Routine-level config; same fields as :class:`BraTSPedImageH5Config`."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> BraTSPedH5RoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


class BraTSPedH5RoutineEngine:
    def __init__(self, cfg: BraTSPedH5RoutineConfig) -> None:
        self.cfg = cfg
        self._inner = BraTSPedImageH5Converter(cfg)

    def run(self) -> Path:
        return self._inner.run()

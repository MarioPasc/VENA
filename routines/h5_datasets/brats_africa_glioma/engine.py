"""Thin routine wrapper around the BraTS-Africa converter for the Glioma subset."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.data.h5.brats_africa.image_domain import (
    BraTSAfricaImageH5Config,
    BraTSAfricaImageH5Converter,
)

logger = logging.getLogger(__name__)


class BraTSAfricaGliomaH5RoutineConfig(BraTSAfricaImageH5Config):
    """Routine-level config; pinned to ``cohort_name='brats_africa_glioma'``."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> BraTSAfricaGliomaH5RoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("cohort_name", "brats_africa_glioma")
        cfg = cls.model_validate(raw)
        if cfg.cohort_name != "brats_africa_glioma":
            raise ValueError(
                f"this routine handles 'brats_africa_glioma' only; got {cfg.cohort_name!r}"
            )
        return cfg


class BraTSAfricaGliomaH5RoutineEngine:
    def __init__(self, cfg: BraTSAfricaGliomaH5RoutineConfig) -> None:
        self.cfg = cfg
        self._inner = BraTSAfricaImageH5Converter(cfg)

    def run(self) -> Path:
        return self._inner.run()

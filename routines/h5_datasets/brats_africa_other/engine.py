"""Thin routine wrapper around the BraTS-Africa converter for the OtherNeoplasms subset."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.data.h5.brats_africa.image_domain import (
    BraTSAfricaImageH5Config,
    BraTSAfricaImageH5Converter,
)

logger = logging.getLogger(__name__)


class BraTSAfricaOtherH5RoutineConfig(BraTSAfricaImageH5Config):
    """Routine-level config; pinned to ``cohort_name='brats_africa_other'``."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> BraTSAfricaOtherH5RoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("cohort_name", "brats_africa_other")
        cfg = cls.model_validate(raw)
        if cfg.cohort_name != "brats_africa_other":
            raise ValueError(
                f"this routine handles 'brats_africa_other' only; got {cfg.cohort_name!r}"
            )
        return cfg


class BraTSAfricaOtherH5RoutineEngine:
    def __init__(self, cfg: BraTSAfricaOtherH5RoutineConfig) -> None:
        self.cfg = cfg
        self._inner = BraTSAfricaImageH5Converter(cfg)

    def run(self) -> Path:
        return self._inner.run()

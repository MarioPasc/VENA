"""Thin routine wrapper around the UCSF-PDGM image-domain H5 converter.

Loads a YAML config into :class:`UCSFPDGMImageH5Config` and delegates to the
library implementation in ``src/vena/data/h5/ucsf_pdgm/image_domain``. Keeps
the routine itself free of business logic so the converter remains importable
and unit-testable in isolation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.data.h5.ucsf_pdgm.image_domain import (
    UCSFPDGMImageH5Config,
    UCSFPDGMImageH5Converter,
)

logger = logging.getLogger(__name__)


class UCSFPDGMH5RoutineConfig(UCSFPDGMImageH5Config):
    """Routine-level config; identical to the library config for now."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> UCSFPDGMH5RoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


class UCSFPDGMH5RoutineEngine:
    """Routine entrypoint that holds the library converter."""

    def __init__(self, cfg: UCSFPDGMH5RoutineConfig) -> None:
        self.cfg = cfg
        self._inner = UCSFPDGMImageH5Converter(cfg)

    def run(self) -> Path:
        return self._inner.run()

"""Thin routine wrapper around the IvyGAP image-domain H5 converter.

Loads a YAML config into :class:`IvyGAPImageH5Config` and delegates to the
library implementation in ``src/vena/data/h5/ivy_gap/image_domain``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.data.h5.ivy_gap.image_domain import (
    IvyGAPImageH5Config,
    IvyGAPImageH5Converter,
)

logger = logging.getLogger(__name__)


class IvyGAPH5RoutineConfig(IvyGAPImageH5Config):
    """Routine-level config; identical to the library config."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> IvyGAPH5RoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


class IvyGAPH5RoutineEngine:
    """Routine entrypoint that holds the library converter."""

    def __init__(self, cfg: IvyGAPH5RoutineConfig) -> None:
        self.cfg = cfg
        self._inner = IvyGAPImageH5Converter(cfg)

    def run(self) -> Path:
        return self._inner.run()

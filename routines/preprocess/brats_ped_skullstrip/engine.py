"""Thin routine wrapper around :class:`HDBETSkullStripRunner`.

Produces a mirror of the BraTS-PED source tree under ``dest_root`` with each
modality multiplied by the HD-BET-derived brain mask and a per-patient
``BraTS-PED-NNNNN-NNN-brain_mask.nii.gz`` artifact alongside.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.preprocess.hd_bet import HDBETSkullStripConfig, HDBETSkullStripRunner

logger = logging.getLogger(__name__)


class BraTSPedSkullStripRoutineConfig(HDBETSkullStripConfig):
    """Routine-level config; same fields as :class:`HDBETSkullStripConfig`."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> BraTSPedSkullStripRoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


class BraTSPedSkullStripRoutineEngine:
    def __init__(self, cfg: BraTSPedSkullStripRoutineConfig) -> None:
        self.cfg = cfg
        self._inner = HDBETSkullStripRunner(cfg)

    def run(self) -> Path:
        return self._inner.run()

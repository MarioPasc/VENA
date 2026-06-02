"""Thin routine wrapper around :class:`HDBETSkullStripRunner` for REMBRANDT.

Produces a mirror of the REMBRANDT source tree under ``dest_root`` with each
modality multiplied by the HD-BET-derived brain mask and a per-patient
``<pid>-brain_mask.nii.gz`` artifact alongside. The GlistrBoost tumour
segmentation is carried through unchanged.

REMBRANDT-specific configuration (vs the BraTS-PED routine):

* ``patient_dir_regex`` matches ``900-00-XXXX_YYYY.MM.DD`` and
  ``HFXXXX_YYYY.MM.DD`` directory names.
* ``modality_filename_template = "{pid}_{suffix}_LPS_rSRI.nii.gz"`` to honour
  the CBICA preprocessing naming convention.
* ``seg_suffix = "GlistrBoost_out"`` and
  ``seg_filename_template = "{pid}_{seg_suffix}.nii.gz"`` (no ``_LPS_rSRI``
  infix on the seg file).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vena.preprocess.hd_bet import HDBETSkullStripConfig, HDBETSkullStripRunner

logger = logging.getLogger(__name__)


class REMBRANDTSkullStripRoutineConfig(HDBETSkullStripConfig):
    """Routine-level config; same fields as :class:`HDBETSkullStripConfig`.

    The REMBRANDT YAMLs simply override the four template/regex defaults so the
    runner discovers and rewrites the right files. No code differences.
    """

    @classmethod
    def from_yaml(cls, path: Path | str) -> REMBRANDTSkullStripRoutineConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


class REMBRANDTSkullStripRoutineEngine:
    def __init__(self, cfg: REMBRANDTSkullStripRoutineConfig) -> None:
        self.cfg = cfg
        self._inner = HDBETSkullStripRunner(cfg)

    def run(self) -> Path:
        return self._inner.run()

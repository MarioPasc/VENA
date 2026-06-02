"""REMBRANDT (CBICA-preprocessed) cohort loader.

Source layout (after the symlink-staging step that flattens the original
``Patients_X_Y/`` batch directories)::

    <source_root>/<pid>_<YYYY.MM.DD>/<pid>_<YYYY.MM.DD>_{t1,t1ce,t2,flair}_LPS_rSRI.nii.gz
    <source_root>/<pid>_<YYYY.MM.DD>/<pid>_<YYYY.MM.DD>_GlistrBoost_out.nii.gz

Patient IDs follow one of two conventions:

* ``900-00-XXXX_YYYY.MM.DD`` (NIH / TCIA-style portal IDs, 20 patients)
* ``HFXXXX_YYYY.MM.DD`` (Henry Ford Hospital legacy IDs, 44 patients)

The data is **registered to SRI24** (1 mm iso, LPS) but **NOT skull-stripped**;
the converter upstream of this reader expects an already-skull-stripped tree
produced by ``routines/preprocess/rembrandt_skullstrip``. Tumour segmentation
comes from the CBICA GLISTRboost pipeline (BraTS-2021 labels ``{0, 1, 2, 4}``).

REMBRANDT is cross-sectional: every directory carries exactly one timepoint,
encoded in the directory name. The patient_id used by the reader is the full
directory name (subject id + date), mirroring the BraTS convention of using
the full per-session string as the cohort-unique ID.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vena.data.cohort import register_cohort

from .shared.exceptions import ModalityNotFoundError, PatientNotFoundError
from .shared.io import NiftiVolume, load_nii

logger = logging.getLogger(__name__)


Modality = Literal["t1pre", "t1c", "t2", "flair"]

# H5 modality slug → REMBRANDT filename infix (between the date and the
# trailing ``_LPS_rSRI.nii.gz``).
_MODALITY_SUFFIX: dict[str, str] = {
    "t1pre": "t1",
    "t1c": "t1ce",
    "t2": "t2",
    "flair": "flair",
}

_SEG_SUFFIX = "GlistrBoost_out"

# Match both ID families: ``900-00-XXXX_YYYY.MM.DD`` and ``HFXXXX_YYYY.MM.DD``.
_PATIENT_DIR_RE = re.compile(r"^(900-00-\d+|HF\d+)_\d{4}\.\d{2}\.\d{2}$")


@dataclass(frozen=True)
class REMBRANDTPatient:
    """A single REMBRANDT patient/session handle (cross-sectional).

    Attributes
    ----------
    patient_id : str
        Full session identifier (e.g. ``900-00-5303_2005.03.24`` or
        ``HF1318_1994.04.23``). Includes the date suffix because that is the
        on-disk directory name and the CBICA-canonical identifier.
    root : Path
        Absolute path to the per-session directory containing the four
        modalities + the GlistrBoost tumour segmentation.
    metadata : dict[str, Any]
        Optional metadata; empty by default. Present for
        :class:`vena.data.cohort.CohortPatient` protocol conformance.
    """

    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@register_cohort(
    "rembrandt",
    pathology="glioma",
    metadata={
        "release": "REMBRANDT (CBICA-preprocessed; LPS / SRI24 1 mm iso)",
        "spacing_mm": (1.0, 1.0, 1.0),
        "atlas": "SRI24",
        "label_system": "BraTS2021",
    },
)
class REMBRANDTDataset:
    """REMBRANDT cohort reader.

    ``source_root`` must be a flat directory containing the per-session
    subdirectories (typically produced by
    ``scripts/prepare_rembrandt_source.sh`` which symlinks the original batch
    folders into one place and renames any incomplete session aside).
    """

    def __init__(self, source_root: Path | str) -> None:
        self.source_root = Path(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source_root does not exist: {self.source_root}")
        self._patients = self._discover_patients()
        self._index_by_id = {p.patient_id: i for i, p in enumerate(self._patients)}
        logger.info(
            "REMBRANDT (%s): discovered %d patient(s)",
            self.source_root.name,
            len(self._patients),
        )

    def _discover_patients(self) -> list[REMBRANDTPatient]:
        patients: list[REMBRANDTPatient] = []
        for d in sorted(self.source_root.iterdir()):
            if not d.is_dir():
                continue
            if _PATIENT_DIR_RE.match(d.name) is None:
                continue
            patients.append(REMBRANDTPatient(patient_id=d.name, root=d))
        return patients

    # ----- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._patients)

    def __iter__(self) -> Iterator[REMBRANDTPatient]:
        return iter(self._patients)

    def __getitem__(self, key: int | str) -> REMBRANDTPatient:
        if isinstance(key, int):
            return self._patients[key]
        if key in self._index_by_id:
            return self._patients[self._index_by_id[key]]
        raise PatientNotFoundError(f"Unknown REMBRANDT patient: {key}")

    def ids(self) -> list[str]:
        return [p.patient_id for p in self._patients]

    # ----- modality access ----------------------------------------------------

    @staticmethod
    def _modality_path(p: REMBRANDTPatient, suffix: str) -> Path:
        # Filename pattern: <pid>_<suffix>_LPS_rSRI.nii.gz
        return p.root / f"{p.patient_id}_{suffix}_LPS_rSRI.nii.gz"

    @staticmethod
    def _seg_path(p: REMBRANDTPatient) -> Path:
        # Filename pattern: <pid>_GlistrBoost_out.nii.gz (no LPS_rSRI infix).
        return p.root / f"{p.patient_id}_{_SEG_SUFFIX}.nii.gz"

    def load_modality(self, p: REMBRANDTPatient, name: Modality) -> NiftiVolume:
        """Load one MR modality for the patient.

        Raises
        ------
        ModalityNotFoundError
            If the file does not exist on disk.
        """
        if name not in _MODALITY_SUFFIX:
            raise ModalityNotFoundError(f"Unknown modality: {name!r}")
        path = self._modality_path(p, _MODALITY_SUFFIX[name])
        if not path.exists():
            raise ModalityNotFoundError(f"Modality {name} missing for {p.patient_id}: {path}")
        return load_nii(path)

    def load_tumor_seg(self, p: REMBRANDTPatient) -> NiftiVolume:
        """Load the GlistrBoost tumour segmentation (BraTS-2021 labels)."""
        path = self._seg_path(p)
        if not path.exists():
            raise ModalityNotFoundError(f"Tumour segmentation missing for {p.patient_id}: {path}")
        return load_nii(path)

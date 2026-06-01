"""BraTS-PED 2024 cohort loader (pediatric high-grade gliomas).

The released set sits at::

    <source_root>/BraTS-PED-NNNNN-TTT/BraTS-PED-NNNNN-TTT-{t1n,t1c,t2w,t2f,seg}.nii.gz

It follows the BraTS-2023 file convention (suffix slugs ``t1n``, ``t1c``,
``t2w``, ``t2f``, ``seg``) and the standard BraTS preprocessing — SRI24 1 mm
iso, BraTS-2023 tumour labels ``{0, 1, 2, 3}``. Unlike adult BraTS releases,
the source data is **defaced only**, not skull-stripped: the converter
upstream of this reader expects an already-skull-stripped tree (e.g. produced
by ``routines/preprocess/brats_ped_skullstrip``).

Patient IDs are ``BraTS-PED-NNNNN-TTT`` (cross-sectional; the trailing
``-TTT`` is the only timepoint in the released set).
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

# H5 modality slug → BraTS file suffix.
_MODALITY_SUFFIX: dict[str, str] = {
    "t1pre": "t1n",
    "t1c": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}

_SEG_SUFFIX = "seg"

# Patient directory pattern: BraTS-PED-NNNNN-TTT (TTT = "000" in the released set).
_PATIENT_DIR_RE = re.compile(r"^BraTS-PED-(\d+)-(\d+)$")


@dataclass(frozen=True)
class BraTSPedPatient:
    """A single BraTS-PED patient handle (cross-sectional).

    Attributes
    ----------
    patient_id : str
        Full BraTS ID including the timepoint suffix (e.g. ``BraTS-PED-00043-000``).
    root : Path
        Absolute path to the per-patient directory containing the five NIfTI
        files (t1n, t1c, t2w, t2f, seg).
    metadata : dict[str, Any]
        Optional metadata; empty by default. Present for
        :class:`vena.data.cohort.CohortPatient` protocol conformance.
    """

    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@register_cohort(
    "brats_ped",
    pathology="glioma",
    metadata={
        "release": "BraTS-PED 2024 Challenge — Training set (pediatric HGG)",
        "spacing_mm": (1.0, 1.0, 1.0),
        "atlas": "SRI24",
        "label_system": "BraTS2023",
    },
)
class BraTSPedDataset:
    """BraTS-PED 2024 pediatric glioma cohort.

    ``source_root`` must point at a directory holding the per-patient subdirs
    (defaced source or HD-BET-stripped mirror).
    """

    def __init__(self, source_root: Path | str) -> None:
        self.source_root = Path(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source_root does not exist: {self.source_root}")
        self._patients = self._discover_patients()
        self._index_by_id = {p.patient_id: i for i, p in enumerate(self._patients)}
        logger.info(
            "BraTS-PED (%s): discovered %d patient(s)",
            self.source_root.name,
            len(self._patients),
        )

    def _discover_patients(self) -> list[BraTSPedPatient]:
        patients: list[BraTSPedPatient] = []
        for d in sorted(self.source_root.iterdir()):
            if not d.is_dir():
                continue
            if _PATIENT_DIR_RE.match(d.name) is None:
                continue
            patients.append(BraTSPedPatient(patient_id=d.name, root=d))
        return patients

    # ----- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._patients)

    def __iter__(self) -> Iterator[BraTSPedPatient]:
        return iter(self._patients)

    def __getitem__(self, key: int | str) -> BraTSPedPatient:
        if isinstance(key, int):
            return self._patients[key]
        if key in self._index_by_id:
            return self._patients[self._index_by_id[key]]
        raise PatientNotFoundError(f"Unknown BraTS-PED patient: {key}")

    def ids(self) -> list[str]:
        return [p.patient_id for p in self._patients]

    # ----- modality access ----------------------------------------------------

    @staticmethod
    def _modality_path(p: BraTSPedPatient, suffix: str) -> Path:
        return p.root / f"{p.patient_id}-{suffix}.nii.gz"

    def load_modality(self, p: BraTSPedPatient, name: Modality) -> NiftiVolume:
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

    def load_tumor_seg(self, p: BraTSPedPatient) -> NiftiVolume:
        """Load the BraTS-2023 tumour segmentation (labels {0, 1, 2, 3})."""
        path = self._modality_path(p, _SEG_SUFFIX)
        if not path.exists():
            raise ModalityNotFoundError(f"Tumour segmentation missing for {p.patient_id}: {path}")
        return load_nii(path)

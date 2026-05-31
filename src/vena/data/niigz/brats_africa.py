"""BraTS-Africa (SSA) cohort loader.

Two subsets are released at the top level of the data root:

    BraTS-Africa/
        95_Glioma/<id>/<id>-{t1n,t1c,t2w,t2f,seg}.nii.gz
        51_OtherNeoplasms/<id>/<id>-{t1n,t1c,t2w,t2f,seg}.nii.gz

Both follow the BraTS-2023 file convention (suffix slugs t1n/t1c/t2w/t2f/seg)
and the standard BraTS preprocessing — SRI24 1 mm iso, skull-stripped, per-
modality z-score normalised within the brain mask, BraTS-2023 tumour labels
``{0, 1, 2, 3}``. Patient IDs are ``BraTS-SSA-NNNNN-000`` (cross-sectional;
the trailing ``-000`` is the only timepoint in the released set).

This module exposes one dataset class, :class:`BraTSAfricaDataset`, that takes
``source_root`` pointing directly at one of the two subset subdirectories. The
class is registered twice under separate cohort names — ``brats_africa_glioma``
and ``brats_africa_other`` — so that each pathology can carry a distinct
``Pathology`` label in the registry. The actual subset is implied by the
caller's ``source_root``.
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

# Patient directory pattern: BraTS-SSA-NNNNN-TTT (TTT = "000" in the released set).
_PATIENT_DIR_RE = re.compile(r"^BraTS-SSA-(\d+)-(\d+)$")


@dataclass(frozen=True)
class BraTSAfricaPatient:
    """A single BraTS-Africa patient handle (cross-sectional).

    Attributes
    ----------
    patient_id : str
        Full BraTS ID including the timepoint suffix (e.g. ``BraTS-SSA-00002-000``).
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


class _BraTSAfricaBase:
    """Shared discovery + IO for the two BraTS-Africa subsets.

    Concrete subclasses :class:`BraTSAfricaGliomaDataset` and
    :class:`BraTSAfricaOtherDataset` only add the ``@register_cohort`` decoration.
    """

    def __init__(self, source_root: Path | str) -> None:
        self.source_root = Path(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source_root does not exist: {self.source_root}")
        self._patients = self._discover_patients()
        self._index_by_id = {p.patient_id: i for i, p in enumerate(self._patients)}
        logger.info(
            "BraTS-Africa (%s): discovered %d patient(s)",
            self.source_root.name,
            len(self._patients),
        )

    def _discover_patients(self) -> list[BraTSAfricaPatient]:
        patients: list[BraTSAfricaPatient] = []
        for d in sorted(self.source_root.iterdir()):
            if not d.is_dir():
                continue
            if _PATIENT_DIR_RE.match(d.name) is None:
                continue
            patients.append(BraTSAfricaPatient(patient_id=d.name, root=d))
        return patients

    # ----- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._patients)

    def __iter__(self) -> Iterator[BraTSAfricaPatient]:
        return iter(self._patients)

    def __getitem__(self, key: int | str) -> BraTSAfricaPatient:
        if isinstance(key, int):
            return self._patients[key]
        if key in self._index_by_id:
            return self._patients[self._index_by_id[key]]
        raise PatientNotFoundError(f"Unknown BraTS-Africa patient: {key}")

    def ids(self) -> list[str]:
        return [p.patient_id for p in self._patients]

    # ----- modality access ----------------------------------------------------

    @staticmethod
    def _modality_path(p: BraTSAfricaPatient, suffix: str) -> Path:
        return p.root / f"{p.patient_id}-{suffix}.nii.gz"

    def load_modality(self, p: BraTSAfricaPatient, name: Modality) -> NiftiVolume:
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

    def load_tumor_seg(self, p: BraTSAfricaPatient) -> NiftiVolume:
        """Load the BraTS-2023 tumour segmentation (labels {0, 1, 2, 3})."""
        path = self._modality_path(p, _SEG_SUFFIX)
        if not path.exists():
            raise ModalityNotFoundError(f"Tumour segmentation missing for {p.patient_id}: {path}")
        return load_nii(path)


@register_cohort(
    "brats_africa_glioma",
    pathology="glioma",
    metadata={
        "release": "BraTS-Africa (SSA) 2023 — 95_Glioma subset",
        "spacing_mm": (1.0, 1.0, 1.0),
        "atlas": "SRI24",
        "label_system": "BraTS2023",
    },
)
class BraTSAfricaGliomaDataset(_BraTSAfricaBase):
    """BraTS-Africa glioma subset (95 patients).

    ``source_root`` must point at ``.../BraTS-Africa/95_Glioma/``.
    """


@register_cohort(
    "brats_africa_other",
    pathology="other",
    metadata={
        "release": "BraTS-Africa (SSA) 2023 — 51_OtherNeoplasms subset",
        "spacing_mm": (1.0, 1.0, 1.0),
        "atlas": "SRI24",
        "label_system": "BraTS2023",
    },
)
class BraTSAfricaOtherDataset(_BraTSAfricaBase):
    """BraTS-Africa other-neoplasms subset (51 patients).

    ``source_root`` must point at ``.../BraTS-Africa/51_OtherNeoplasms/``.
    """

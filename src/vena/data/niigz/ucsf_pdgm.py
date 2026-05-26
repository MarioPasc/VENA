"""UCSF-PDGM cohort loader for the NIfTI source release.

Source layout (one directory per patient under ``source_root``):

    UCSF-PDGM-{NNNN}_nifti/
        UCSF-PDGM-{NNNN}_T1.nii.gz
        UCSF-PDGM-{NNNN}_T1_bias.nii.gz
        UCSF-PDGM-{NNNN}_T1c.nii.gz
        UCSF-PDGM-{NNNN}_T1c_bias.nii.gz
        UCSF-PDGM-{NNNN}_T2.nii.gz            ...
        UCSF-PDGM-{NNNN}_FLAIR.nii.gz         ...
        UCSF-PDGM-{NNNN}_SWI.nii.gz           ...
        UCSF-PDGM-{NNNN}_DWI.nii.gz           ...
        UCSF-PDGM-{NNNN}_ADC.nii.gz   (mm^2/s, ~[0, 5e-3])
        UCSF-PDGM-{NNNN}_ASL.nii.gz   (CBF, ~ml/100g/min, single-PLD pCASL)
        UCSF-PDGM-{NNNN}_brain_segmentation.nii.gz
        UCSF-PDGM-{NNNN}_brain_parenchyma_segmentation.nii.gz
        UCSF-PDGM-{NNNN}_tumor_segmentation.nii.gz

All modalities share the T1 affine (verified at 1 mm isotropic, LPS), so the
loader returns volumes without resampling. The associated metadata CSV uses a
3-digit (no leading zero) ID format; the join helper zero-pads to 4 digits.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .shared.exceptions import ModalityNotFoundError, PatientNotFoundError
from .shared.io import NiftiVolume, load_nii

logger = logging.getLogger(__name__)


Modality = Literal[
    "T1",
    "T1_bias",
    "T1c",
    "T1c_bias",
    "T2",
    "T2_bias",
    "FLAIR",
    "FLAIR_bias",
    "SWI",
    "SWI_bias",
    "DWI",
    "DWI_bias",
    "ADC",
    "ASL",
]

# File suffix attached after the patient ID for each modality, e.g.
# "{PID}_T1.nii.gz". Stored once here so adding a new modality is a one-line change.
_MODALITY_SUFFIX: dict[str, str] = {
    "T1": "T1",
    "T1_bias": "T1_bias",
    "T1c": "T1c",
    "T1c_bias": "T1c_bias",
    "T2": "T2",
    "T2_bias": "T2_bias",
    "FLAIR": "FLAIR",
    "FLAIR_bias": "FLAIR_bias",
    "SWI": "SWI",
    "SWI_bias": "SWI_bias",
    "DWI": "DWI",
    "DWI_bias": "DWI_bias",
    "ADC": "ADC",
    "ASL": "ASL",
}

_BRAIN_MASK_SUFFIX = "brain_segmentation"
_BRAIN_PARENCHYMA_SUFFIX = "brain_parenchyma_segmentation"
_TUMOR_SEG_SUFFIX = "tumor_segmentation"

_PATIENT_DIR_RE = re.compile(r"^UCSF-PDGM-(\d{4})_nifti$")


@dataclass(frozen=True)
class UCSFPDGMPatient:
    """A single UCSF-PDGM patient handle.

    The dataclass is intentionally lightweight: it carries paths and metadata but
    no voxel data. Voxel data is fetched on demand through
    :class:`UCSFPDGMDataset` accessors so a `sample(10)` call costs no I/O until
    a modality is actually requested.
    """

    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class UCSFPDGMDataset:
    """Index of the UCSF-PDGM source cohort.

    Parameters
    ----------
    source_root
        Directory containing ``UCSF-PDGM-XXXX_nifti/`` subdirectories.
    metadata_csv
        Optional path to ``UCSF-PDGM-metadata_v5.csv``. When provided, each
        patient's row is attached to the :class:`UCSFPDGMPatient` as a dict.

    Raises
    ------
    FileNotFoundError
        If ``source_root`` does not exist.
    """

    def __init__(
        self,
        source_root: Path | str,
        metadata_csv: Path | str | None = None,
    ) -> None:
        self.source_root = Path(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source_root does not exist: {self.source_root}")
        self._metadata = self._load_metadata(metadata_csv)
        self._patients = self._discover_patients()
        self._index_by_id = {p.patient_id: i for i, p in enumerate(self._patients)}
        logger.info("UCSF-PDGM: discovered %d patient directories", len(self._patients))

    def _load_metadata(self, metadata_csv: Path | str | None) -> dict[str, dict[str, Any]]:
        if metadata_csv is None:
            return {}
        csv_path = Path(metadata_csv)
        if not csv_path.exists():
            logger.warning("Metadata CSV does not exist: %s (continuing without)", csv_path)
            return {}
        df = pd.read_csv(csv_path)
        # CSV IDs look like "UCSF-PDGM-004"; directory IDs are "UCSF-PDGM-0004".
        # Zero-pad the trailing integer so we can join on the 4-digit form.
        out: dict[str, dict[str, Any]] = {}
        id_col = df.columns[0]
        for _, row in df.iterrows():
            raw = str(row[id_col])
            m = re.match(r"^UCSF-PDGM-(\d+)$", raw.strip())
            if m is None:
                continue
            padded = f"UCSF-PDGM-{int(m.group(1)):04d}"
            out[padded] = row.to_dict()
        return out

    def _discover_patients(self) -> list[UCSFPDGMPatient]:
        patients: list[UCSFPDGMPatient] = []
        for d in sorted(self.source_root.iterdir()):
            if not d.is_dir():
                continue
            m = _PATIENT_DIR_RE.match(d.name)
            if m is None:
                continue
            pid = f"UCSF-PDGM-{m.group(1)}"
            patients.append(
                UCSFPDGMPatient(
                    patient_id=pid,
                    root=d,
                    metadata=self._metadata.get(pid, {}),
                )
            )
        return patients

    # ----- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._patients)

    def __iter__(self) -> Iterator[UCSFPDGMPatient]:
        return iter(self._patients)

    def __getitem__(self, key: int | str) -> UCSFPDGMPatient:
        if isinstance(key, int):
            return self._patients[key]
        if key in self._index_by_id:
            return self._patients[self._index_by_id[key]]
        raise PatientNotFoundError(f"Unknown UCSF-PDGM patient: {key}")

    def ids(self) -> list[str]:
        return [p.patient_id for p in self._patients]

    def sample(self, n: int, *, seed: int) -> list[UCSFPDGMPatient]:
        """Deterministically draw ``n`` patients without replacement."""
        if n > len(self._patients):
            raise ValueError(f"Requested {n} patients but cohort has {len(self._patients)}")
        rng = random.Random(seed)
        return rng.sample(self._patients, k=n)

    # ----- modality access ----------------------------------------------------

    def _modality_path(self, p: UCSFPDGMPatient, suffix: str) -> Path:
        return p.root / f"{p.patient_id}_{suffix}.nii.gz"

    def load_modality(self, p: UCSFPDGMPatient, name: Modality) -> NiftiVolume:
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

    def load_brain_mask(self, p: UCSFPDGMPatient) -> NiftiVolume:
        """Load the binary brain mask. Values are float-encoded near 1.0."""
        path = self._modality_path(p, _BRAIN_MASK_SUFFIX)
        if not path.exists():
            raise ModalityNotFoundError(f"Brain mask missing for {p.patient_id}: {path}")
        return load_nii(path)

    def load_brain_parenchyma_mask(self, p: UCSFPDGMPatient) -> NiftiVolume:
        path = self._modality_path(p, _BRAIN_PARENCHYMA_SUFFIX)
        if not path.exists():
            raise ModalityNotFoundError(f"Brain parenchyma mask missing for {p.patient_id}: {path}")
        return load_nii(path)

    def load_tumor_seg(self, p: UCSFPDGMPatient) -> NiftiVolume:
        """Load the BraTS-style tumour segmentation (labels {0, 1, 2, 4})."""
        path = self._modality_path(p, _TUMOR_SEG_SUFFIX)
        if not path.exists():
            raise ModalityNotFoundError(f"Tumor segmentation missing for {p.patient_id}: {path}")
        return load_nii(path)

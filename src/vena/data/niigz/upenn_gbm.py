"""UPENN-GBM cohort loader for the NIfTI source release.

Source layout (UPENN-GBM v2.0 release on TCIA; preoperative-only — the
trailing ``_11`` is the preoperative timepoint suffix used by the release)::

    <source_root>/                                  # images_structural/
        UPENN-GBM-NNNNN_11/
            UPENN-GBM-NNNNN_11_T1.nii.gz
            UPENN-GBM-NNNNN_11_T1GD.nii.gz
            UPENN-GBM-NNNNN_11_T2.nii.gz
            UPENN-GBM-NNNNN_11_FLAIR.nii.gz
    <source_root>/../images_segm/
        UPENN-GBM-NNNNN_11_segm.nii.gz              # manual (gold; 147 patients)
    <source_root>/../automated_segm/
        UPENN-GBM-NNNNN_11_automated_approx_segm.nii.gz   # auto (611 patients)

All volumes share an LPS affine at 1 mm isotropic ``(240, 240, 155)``
(SRI24 atlas, BraTS-standard); the release is already skull-stripped.

Segmentations follow the BraTS-2021 label system ``{0, 1, 2, 4}``. The reader
prefers the manual seg when present and falls back to the automated seg;
the choice is recorded per patient in :attr:`UPENNGBMPatient.metadata`
under ``seg_source``.

The cohort is cross-sectional (one timepoint per patient). 671 patients
have structural images; 60 of those have no segmentation at all and are
excluded by the discovery pass — the converter never sees them.

Cross-cohort deduplication against BraTS-GLI uses the
``metadata/brats21_id`` field stored in the image-H5 (same contract as
UCSF-PDGM, see :mod:`vena.data.h5.upenn_gbm.image_domain.manifest`). The
metadata join expects the lookup CSV built by
``scripts/preprocess/build_upenn_gbm_brats21_lookup.py`` (columns
``patient_id, brats21_id, brats21_data_collection``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from vena.data.cohort import register_cohort

from .shared.exceptions import ModalityNotFoundError, PatientNotFoundError
from .shared.io import NiftiVolume, load_nii

logger = logging.getLogger(__name__)


Modality = Literal["t1pre", "t1c", "t2", "flair"]

# H5 modality slug → BraTS file suffix used by the UPenn release.
_MODALITY_SUFFIX: dict[str, str] = {
    "t1pre": "T1",
    "t1c": "T1GD",
    "t2": "T2",
    "flair": "FLAIR",
}

_MANUAL_SEG_SUFFIX = "segm"
_AUTO_SEG_SUFFIX = "automated_approx_segm"

_PATIENT_DIR_RE = re.compile(r"^UPENN-GBM-(\d{5})_(\d{2})$")


@dataclass(frozen=True)
class UPENNGBMPatient:
    """A single UPENN-GBM patient handle.

    The dataclass is intentionally lightweight: it carries paths and metadata
    but no voxel data. Voxel data is fetched on demand through
    :class:`UPENNGBMDataset` accessors so a ``sample(10)`` call costs no I/O
    until a modality is actually requested.

    Attributes
    ----------
    patient_id : str
        Full directory name with timepoint suffix, e.g. ``UPENN-GBM-00001_11``.
    root : Path
        Absolute path to the structural image directory.
    metadata : dict[str, Any]
        Optional metadata joined from the BraTS-21 lookup CSV; carries
        ``brats21_id`` (possibly empty), ``brats21_data_collection``, and
        ``seg_source`` ∈ ``{"manual", "automated"}``.
    """

    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@register_cohort(
    "upenn_gbm",
    pathology="glioma",
    metadata={
        "release": "UPENN-GBM v2.0",
        "spacing_mm": (1.0, 1.0, 1.0),
        "atlas": "SRI24",
        "label_system": "BraTS2021",
    },
)
class UPENNGBMDataset:
    """Index of the UPENN-GBM source cohort.

    Implements :class:`vena.data.cohort.CohortProtocol` structurally
    (registered via :func:`vena.data.cohort.register_cohort`).

    Patients without ANY segmentation (manual or automated) are dropped
    during discovery — the converter never iterates them, so they do not
    enter ``ids``/``patients/keys``/``splits/*``.

    Parameters
    ----------
    source_root
        Directory containing ``UPENN-GBM-NNNNN_11/`` subdirectories
        (i.e. ``<release>/NIfTI-files/images_structural``).
    metadata_csv
        Optional path to ``UPENN-GBM_brats21_lookup_v1.csv`` produced by
        ``scripts/preprocess/build_upenn_gbm_brats21_lookup.py``. When
        provided, each patient's BraTS-21 mapping (if any) is attached to
        the :class:`UPENNGBMPatient` as a dict.

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
        self._seg_root_manual = self.source_root.parent / "images_segm"
        self._seg_root_auto = self.source_root.parent / "automated_segm"
        self._metadata = self._load_metadata(metadata_csv)
        self._patients = self._discover_patients()
        self._index_by_id = {p.patient_id: i for i, p in enumerate(self._patients)}
        logger.info(
            "UPENN-GBM: discovered %d patient directories with segmentation",
            len(self._patients),
        )

    # ----- metadata ---------------------------------------------------------

    def _load_metadata(self, metadata_csv: Path | str | None) -> dict[str, dict[str, Any]]:
        if metadata_csv is None:
            return {}
        csv_path = Path(metadata_csv)
        if not csv_path.exists():
            logger.warning("Metadata CSV does not exist: %s (continuing without)", csv_path)
            return {}
        df = pd.read_csv(csv_path)
        id_col = "patient_id" if "patient_id" in df.columns else df.columns[0]
        out: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            pid = str(row[id_col]).strip()
            if not pid:
                continue
            out[pid] = row.to_dict()
        return out

    # ----- discovery --------------------------------------------------------

    def _seg_paths(self, pid: str) -> tuple[Path, Path]:
        """Return ``(manual_path, auto_path)`` for a given patient ID."""
        return (
            self._seg_root_manual / f"{pid}_{_MANUAL_SEG_SUFFIX}.nii.gz",
            self._seg_root_auto / f"{pid}_{_AUTO_SEG_SUFFIX}.nii.gz",
        )

    def _resolve_seg(self, pid: str) -> tuple[Path | None, str | None]:
        """Return ``(path, source)`` for a patient. Prefer manual, fall back to auto."""
        manual, auto = self._seg_paths(pid)
        if manual.exists():
            return manual, "manual"
        if auto.exists():
            return auto, "automated"
        return None, None

    def _discover_patients(self) -> list[UPENNGBMPatient]:
        patients: list[UPENNGBMPatient] = []
        n_skipped_no_seg = 0
        for d in sorted(self.source_root.iterdir()):
            if not d.is_dir():
                continue
            if _PATIENT_DIR_RE.match(d.name) is None:
                continue
            pid = d.name
            seg_path, seg_source = self._resolve_seg(pid)
            if seg_path is None:
                n_skipped_no_seg += 1
                continue
            meta = dict(self._metadata.get(pid, {}))
            meta["seg_source"] = seg_source
            patients.append(UPENNGBMPatient(patient_id=pid, root=d, metadata=meta))
        if n_skipped_no_seg:
            logger.warning(
                "UPENN-GBM: dropped %d patient(s) without any tumour segmentation",
                n_skipped_no_seg,
            )
        return patients

    # ----- container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self._patients)

    def __iter__(self) -> Iterator[UPENNGBMPatient]:
        return iter(self._patients)

    def __getitem__(self, key: int | str) -> UPENNGBMPatient:
        if isinstance(key, int):
            return self._patients[key]
        if key in self._index_by_id:
            return self._patients[self._index_by_id[key]]
        raise PatientNotFoundError(f"Unknown UPENN-GBM patient: {key}")

    def ids(self) -> list[str]:
        return [p.patient_id for p in self._patients]

    # ----- modality access --------------------------------------------------

    def _modality_path(self, p: UPENNGBMPatient, suffix: str) -> Path:
        return p.root / f"{p.patient_id}_{suffix}.nii.gz"

    def load_modality(self, p: UPENNGBMPatient, name: Modality) -> NiftiVolume:
        if name not in _MODALITY_SUFFIX:
            raise ModalityNotFoundError(f"Unknown modality: {name!r}")
        path = self._modality_path(p, _MODALITY_SUFFIX[name])
        if not path.exists():
            raise ModalityNotFoundError(f"Modality {name} missing for {p.patient_id}: {path}")
        return load_nii(path)

    def load_tumor_seg(self, p: UPENNGBMPatient) -> NiftiVolume:
        """Load the tumour segmentation (manual where available, otherwise auto)."""
        seg_path, _ = self._resolve_seg(p.patient_id)
        if seg_path is None:
            raise ModalityNotFoundError(f"Tumour seg missing for {p.patient_id}")
        return load_nii(seg_path)

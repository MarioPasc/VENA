"""IvyGAP cohort loader (SRI24 atlas preprocessing).

Source layout (one directory per patient under ``source_root``):

    1_Images_SRI/CoRegistered_SkullStripped/
        W<N>/
            W<N>_<YYYY.MM.DD>/
                W<N>_<date>_flair_LPS_[N4_][r|r3]_SS.nii.gz
                W<N>_<date>_t1_LPS_[N4_][r|r3]_SS.nii.gz
                W<N>_<date>_t1gd_LPS_[N4_][r|r3]_SS.nii.gz
                W<N>_<date>_t2_LPS_[N4_][r|r3]_SS.nii.gz
    3_Annotations_SRI/
        CWRU/W<N>/W<N>_<date>_CWRU_labels.nii.gz   (31/34 patients)
        UPenn/W<N>/W<N>_<date>_UPenn_labels.nii.gz (34/34 patients)

All volumes share the SRI24 affine (240, 240, 155) at 1 mm isotropic, LPS-
oriented, skull-stripped, raw scanner intensities. Tumour labels follow the
BraTS-2021 convention {0, 1, 2, 4}.

Per-modality filename precedence is ``("_N4_", "_r3_", "_r_")`` — preferred
variants are bias-corrected (W20 only) > 3rd-order registration > standard
co-registration. Mixed variants within a single patient are tolerated; the
loader records the chosen basename in :attr:`IvyGAPPatient.metadata` under
``source_basename_<modality>``.

Tumour mask source is fixed to UPenn (full 34/34 coverage). The CWRU
annotation path is stored in metadata when present but not exposed as the
canonical tumour segmentation.
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

# Map H5 modality slug → source filename infix.
_MODALITY_INFIX: dict[str, str] = {
    "t1pre": "t1",
    "t1c": "t1gd",
    "t2": "t2",
    "flair": "flair",
}

# Filename precedence for the registration / bias-correction variant. Higher
# precedence (earlier) wins when multiple variants of the same modality exist
# in a patient directory.
_VARIANT_PRECEDENCE: tuple[str, ...] = ("_N4_r_SS", "_r3_SS", "_r_SS")

_PATIENT_DIR_RE = re.compile(r"^W(\d+)$")
_SESSION_DIR_RE = re.compile(r"^W\d+_(\d{4}\.\d{2}\.\d{2})$")

# Subdirectory layout within ``source_root``.
_IMAGES_SUBDIR = Path("1_Images_SRI") / "CoRegistered_SkullStripped"
_ANNOT_UPENN_SUBDIR = Path("3_Annotations_SRI") / "UPenn"
_ANNOT_CWRU_SUBDIR = Path("3_Annotations_SRI") / "CWRU"


@dataclass(frozen=True)
class IvyGAPPatient:
    """A single IvyGAP patient handle.

    Attributes
    ----------
    patient_id : str
        Patient identifier (e.g. ``"W1"``). Cross-sectional: one row per patient.
    root : Path
        Absolute path to the per-patient *session* directory containing the
        four modality NIfTI files (e.g. ``.../W1/W1_1996.10.25``).
    metadata : dict[str, Any]
        Per-patient metadata. Always carries ``scan_date`` (string from the
        session directory name) and ``source_basename_<modality>`` for each
        of t1pre / t1c / t2 / flair. ``upenn_seg_path`` and (when available)
        ``cwru_seg_path`` are stored as absolute path strings.
    """

    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@register_cohort(
    "ivy_gap",
    pathology="glioma",
    metadata={
        "release": "IvyGAP-Radiomics-SRI (Multi-Institutional Paired Expert Segmentations)",
        "spacing_mm": (1.0, 1.0, 1.0),
        "atlas": "SRI24",
    },
)
class IvyGAPDataset:
    """Index of the IvyGAP SRI-atlas cohort.

    Implements :class:`vena.data.cohort.CohortProtocol` structurally
    (registered via :func:`vena.data.cohort.register_cohort`).

    Parameters
    ----------
    source_root
        Directory containing ``1_Images_SRI/`` and ``3_Annotations_SRI/`` —
        typically ``.../Multi-Institutional Paired Expert Segmentations SRI
        images-atlas-annotations``.
    tumor_seg_source
        Annotation source used as the canonical tumour mask. Only ``"upenn"``
        is supported today; ``"cwru"`` would drop 3 of 34 patients.

    Raises
    ------
    FileNotFoundError
        If ``source_root`` or the expected subdirectories do not exist.
    """

    def __init__(
        self,
        source_root: Path | str,
        *,
        tumor_seg_source: Literal["upenn"] = "upenn",
    ) -> None:
        self.source_root = Path(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source_root does not exist: {self.source_root}")
        self._images_root = self.source_root / _IMAGES_SUBDIR
        self._upenn_root = self.source_root / _ANNOT_UPENN_SUBDIR
        self._cwru_root = self.source_root / _ANNOT_CWRU_SUBDIR
        if not self._images_root.is_dir():
            raise FileNotFoundError(f"images subdir does not exist: {self._images_root}")
        if not self._upenn_root.is_dir():
            raise FileNotFoundError(f"UPenn annotations subdir does not exist: {self._upenn_root}")
        self._tumor_seg_source: Literal["upenn"] = tumor_seg_source
        self._patients = self._discover_patients()
        self._index_by_id = {p.patient_id: i for i, p in enumerate(self._patients)}
        logger.info("IvyGAP: discovered %d patient(s)", len(self._patients))

    # ----- discovery ---------------------------------------------------------

    def _discover_patients(self) -> list[IvyGAPPatient]:
        patients: list[IvyGAPPatient] = []
        for patient_dir in sorted(self._images_root.iterdir()):
            if not patient_dir.is_dir():
                continue
            m_pid = _PATIENT_DIR_RE.match(patient_dir.name)
            if m_pid is None:
                continue
            patient_id = patient_dir.name  # "W<N>"
            session_dir = self._find_session_dir(patient_dir, patient_id)
            if session_dir is None:
                logger.warning(
                    "IvyGAP: %s has no W<N>_<date> session subdir; skipped",
                    patient_id,
                )
                continue
            scan_date = self._extract_scan_date(session_dir)
            session_stem = session_dir.name  # "W<N>_<YYYY.MM.DD>"
            modality_paths = self._resolve_modality_paths(session_dir, session_stem)
            if modality_paths is None:
                continue  # warning emitted inside helper
            upenn_seg = self._resolve_segmentation(
                self._upenn_root, patient_id, session_stem, "UPenn"
            )
            if upenn_seg is None:
                logger.warning("IvyGAP: %s has no UPenn segmentation; skipped", patient_id)
                continue
            cwru_seg = self._resolve_segmentation(self._cwru_root, patient_id, session_stem, "CWRU")

            metadata: dict[str, Any] = {
                "scan_date": scan_date,
                "session_stem": session_stem,
                "tumor_seg_source": self._tumor_seg_source,
                "upenn_seg_path": str(upenn_seg),
                "cwru_seg_path": "" if cwru_seg is None else str(cwru_seg),
            }
            for slug, path in modality_paths.items():
                metadata[f"source_basename_{slug}"] = path.name
            patients.append(
                IvyGAPPatient(
                    patient_id=patient_id,
                    root=session_dir,
                    metadata=metadata,
                )
            )
        return patients

    @staticmethod
    def _find_session_dir(patient_dir: Path, patient_id: str) -> Path | None:
        candidates = [d for d in sorted(patient_dir.iterdir()) if d.is_dir()]
        for d in candidates:
            if _SESSION_DIR_RE.match(d.name) and d.name.startswith(f"{patient_id}_"):
                return d
        return None

    @staticmethod
    def _extract_scan_date(session_dir: Path) -> str:
        m = _SESSION_DIR_RE.match(session_dir.name)
        return m.group(1) if m else ""

    @staticmethod
    def _resolve_modality_paths(session_dir: Path, session_stem: str) -> dict[str, Path] | None:
        """Pick the highest-precedence variant for each modality."""
        out: dict[str, Path] = {}
        for slug, infix in _MODALITY_INFIX.items():
            chosen: Path | None = None
            for suffix in _VARIANT_PRECEDENCE:
                candidate = session_dir / (f"{session_stem}_{infix}_LPS{suffix}.nii.gz")
                if candidate.exists():
                    chosen = candidate
                    break
            if chosen is None:
                logger.warning(
                    "IvyGAP: %s missing modality %s in %s; patient skipped",
                    session_stem,
                    slug,
                    session_dir,
                )
                return None
            out[slug] = chosen
        return out

    @staticmethod
    def _resolve_segmentation(
        annot_root: Path,
        patient_id: str,
        session_stem: str,
        annotator: str,
    ) -> Path | None:
        candidate = annot_root / patient_id / f"{session_stem}_{annotator}_labels.nii.gz"
        return candidate if candidate.exists() else None

    # ----- container protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self._patients)

    def __iter__(self) -> Iterator[IvyGAPPatient]:
        return iter(self._patients)

    def __getitem__(self, key: int | str) -> IvyGAPPatient:
        if isinstance(key, int):
            return self._patients[key]
        if key in self._index_by_id:
            return self._patients[self._index_by_id[key]]
        raise PatientNotFoundError(f"Unknown IvyGAP patient: {key}")

    def ids(self) -> list[str]:
        return [p.patient_id for p in self._patients]

    # ----- modality access ---------------------------------------------------

    def load_modality(self, p: IvyGAPPatient, name: Modality) -> NiftiVolume:
        """Load one MR modality for the patient.

        Raises
        ------
        ModalityNotFoundError
            If the resolved file does not exist on disk.
        """
        if name not in _MODALITY_INFIX:
            raise ModalityNotFoundError(f"Unknown modality: {name!r}")
        basename = p.metadata.get(f"source_basename_{name}")
        if not basename:
            raise ModalityNotFoundError(f"Modality {name} not resolved for {p.patient_id}")
        path = p.root / basename
        if not path.exists():
            raise ModalityNotFoundError(f"Modality {name} missing for {p.patient_id}: {path}")
        return load_nii(path)

    def load_tumor_seg(self, p: IvyGAPPatient) -> NiftiVolume:
        """Load the UPenn BraTS-style tumour segmentation (labels {0, 1, 2, 4})."""
        upenn_path_str = p.metadata.get("upenn_seg_path", "")
        if not upenn_path_str:
            raise ModalityNotFoundError(f"UPenn tumour segmentation missing for {p.patient_id}")
        upenn_path = Path(upenn_path_str)
        if not upenn_path.exists():
            raise ModalityNotFoundError(
                f"UPenn tumour segmentation missing for {p.patient_id}: {upenn_path}"
            )
        return load_nii(upenn_path)

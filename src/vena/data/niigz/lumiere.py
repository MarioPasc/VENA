"""LUMIERE cohort loader (DeepBraTumIA atlas/skull-strip preprocessing).

Source layout (one directory per patient under ``source_root``):

    LUMIERE/Imaging-v202211/Imaging/
        Patient-NNN/
            week-NNN[-N]/                                 # session
                CT1.nii.gz, T1.nii.gz, T2.nii.gz, FLAIR.nii.gz  (raw — unused)
                DeepBraTumIA-segmentation/atlas/
                    skull_strip/
                        brain_mask.nii.gz
                        ct1_skull_strip.nii.gz
                        flair_skull_strip.nii.gz
                        t1_skull_strip.nii.gz
                        t2_skull_strip.nii.gz
                    segmentation/seg_mask.nii.gz
                HD-GLIO-AUTO-segmentation/...                # alternative seg (unused)

The reader returns paths under the ``DeepBraTumIA-segmentation/atlas/`` tree:
MNI152 1 mm isotropic, shape ``(182, 218, 182)``, skull-stripped, float32. The
DeepBraTumIA seg uses BraTS-2023 labels ``{0, 1, 2, 3}``. The raw per-session
files are anisotropic native-space scans and are *not* used.

Each ``(patient, session)`` pair becomes one independent scan row downstream
(VENA is cross-sectional). Sessions of a patient are kept contiguous so the
CSR ``patients/{offsets, keys}`` layout can be written without re-sorting.
A session is skipped when any required modality or the segmentation is
absent (~2 - 4% of the 638 sessions per ``LUMIERE-datacompleteness.csv``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vena.data.cohort import register_cohort

from .shared.exceptions import ModalityNotFoundError
from .shared.io import NiftiVolume, load_nii

logger = logging.getLogger(__name__)


Modality = Literal["t1pre", "t1c", "t2", "flair"]

# H5 modality slug → basename under DeepBraTumIA-segmentation/atlas/skull_strip/.
_MODALITY_FILENAME: dict[str, str] = {
    "t1pre": "t1_skull_strip.nii.gz",
    "t1c": "ct1_skull_strip.nii.gz",
    "t2": "t2_skull_strip.nii.gz",
    "flair": "flair_skull_strip.nii.gz",
}
_SEG_RELPATH = Path("DeepBraTumIA-segmentation") / "atlas" / "segmentation" / "seg_mask.nii.gz"
_SKULL_STRIP_RELPATH = Path("DeepBraTumIA-segmentation") / "atlas" / "skull_strip"
_BRAIN_MASK_RELPATH = _SKULL_STRIP_RELPATH / "brain_mask.nii.gz"

_PATIENT_DIR_RE = re.compile(r"^Patient-(\d+)$")
_SESSION_DIR_RE = re.compile(r"^week-(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class LUMIERESession:
    """A single LUMIERE session handle (cross-sectional row).

    Attributes
    ----------
    session_id : str
        Full ``Patient-NNN__week-NNN[-N]`` session key (unique per row).
    patient_id : str
        Patient-level identifier (``Patient-NNN``), used for CSR grouping
        and patient-level splits.
    root : Path
        Absolute path to the session directory containing the four raw
        NIfTI files plus the ``DeepBraTumIA-segmentation/`` and
        ``HD-GLIO-AUTO-segmentation/`` subtrees.
    metadata : dict[str, Any]
        Per-session metadata. Always carries ``week`` (int) and ``week_repeat``
        (int suffix or -1 when absent). Other CSV-derived fields are
        attached by the converter at H5 write time.
    """

    session_id: str
    patient_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@register_cohort(
    "lumiere",
    pathology="glioma",
    metadata={
        "release": "LUMIERE v202211 (Suter et al., Sci. Data 2022)",
        "spacing_mm": (1.0, 1.0, 1.0),
        "atlas": "MNI152",
    },
)
class LUMIEREDataset:
    """Index of the LUMIERE longitudinal cohort.

    Implements :class:`vena.data.cohort.CohortProtocol` structurally
    (registered via :func:`vena.data.cohort.register_cohort`).

    Parameters
    ----------
    source_root
        Directory containing the per-patient subdirectories — typically
        ``.../LUMIERE/Imaging-v202211/Imaging``.

    Raises
    ------
    FileNotFoundError
        If ``source_root`` does not exist.
    """

    def __init__(self, source_root: Path | str) -> None:
        self.source_root = Path(source_root)
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source_root does not exist: {self.source_root}")
        self._sessions, self._patient_groups = self._discover()
        logger.info(
            "LUMIERE: discovered %d session(s) from %d patient(s)",
            len(self._sessions),
            len(self._patient_groups),
        )

    # ----- discovery ---------------------------------------------------------

    def _discover(
        self,
    ) -> tuple[list[LUMIERESession], list[tuple[str, list[int]]]]:
        """Discover sessions; build a CSR-ready patient grouping.

        Sessions whose DeepBraTumIA atlas tree is incomplete (missing any of
        the four modalities or the tumour segmentation) are dropped with a
        WARNING. Patient ordering is lexicographic; within a patient, sessions
        are ordered by ``(week, repeat)``.
        """
        raw: list[tuple[str, int, int, str, Path]] = []
        for patient_dir in sorted(self.source_root.iterdir()):
            if not patient_dir.is_dir():
                continue
            m_pid = _PATIENT_DIR_RE.match(patient_dir.name)
            if m_pid is None:
                continue
            patient_id = patient_dir.name
            for session_dir in sorted(patient_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                m_sess = _SESSION_DIR_RE.match(session_dir.name)
                if m_sess is None:
                    continue
                if not self._has_required_files(session_dir):
                    logger.warning(
                        "LUMIERE: %s/%s missing required DeepBraTumIA atlas files; skipped",
                        patient_id,
                        session_dir.name,
                    )
                    continue
                week = int(m_sess.group(1))
                repeat = int(m_sess.group(2)) if m_sess.group(2) is not None else -1
                session_id = f"{patient_id}__{session_dir.name}"
                raw.append((patient_id, week, repeat, session_id, session_dir))

        raw.sort(key=lambda t: (t[0], t[1], t[2]))

        sessions: list[LUMIERESession] = []
        patient_groups: list[tuple[str, list[int]]] = []
        current_patient: str | None = None
        current_indices: list[int] = []

        for i, (pid, week, repeat, session_id, sdir) in enumerate(raw):
            sessions.append(
                LUMIERESession(
                    session_id=session_id,
                    patient_id=pid,
                    root=sdir,
                    metadata={"week": week, "week_repeat": repeat},
                )
            )
            if pid != current_patient:
                if current_patient is not None:
                    patient_groups.append((current_patient, current_indices))
                current_patient = pid
                current_indices = [i]
            else:
                current_indices.append(i)
        if current_patient is not None:
            patient_groups.append((current_patient, current_indices))
        return sessions, patient_groups

    @staticmethod
    def _has_required_files(session_dir: Path) -> bool:
        skull = session_dir / _SKULL_STRIP_RELPATH
        if not skull.is_dir():
            return False
        for fname in _MODALITY_FILENAME.values():
            if not (skull / fname).exists():
                return False
        if not (session_dir / _SEG_RELPATH).exists():
            return False
        return True

    # ----- container protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self._sessions)

    def __iter__(self) -> Iterator[LUMIERESession]:
        return iter(self._sessions)

    def __getitem__(self, index: int) -> LUMIERESession:
        return self._sessions[index]

    def ids(self) -> list[str]:
        """Return all session IDs in discovery order (CSR-compatible)."""
        return [s.session_id for s in self._sessions]

    # ----- cohort-level accessors --------------------------------------------

    def sessions(self) -> list[LUMIERESession]:
        return list(self._sessions)

    def patient_groups(self) -> list[tuple[str, list[int]]]:
        """Return ``(patient_id, [row_indices])`` pairs in patient-sorted order."""
        return list(self._patient_groups)

    def patient_ids(self) -> list[str]:
        return [pid for pid, _ in self._patient_groups]

    # ----- modality access ---------------------------------------------------

    @staticmethod
    def _modality_path(session: LUMIERESession, name: Modality) -> Path:
        return session.root / _SKULL_STRIP_RELPATH / _MODALITY_FILENAME[name]

    def load_modality(self, session: LUMIERESession, name: Modality) -> NiftiVolume:
        """Load one MR modality for the session."""
        if name not in _MODALITY_FILENAME:
            raise ModalityNotFoundError(f"Unknown modality: {name!r}")
        path = self._modality_path(session, name)
        if not path.exists():
            raise ModalityNotFoundError(f"Modality {name} missing for {session.session_id}: {path}")
        return load_nii(path)

    def load_tumor_seg(self, session: LUMIERESession) -> NiftiVolume:
        """Load the DeepBraTumIA atlas-space segmentation (labels {0, 1, 2, 3})."""
        path = session.root / _SEG_RELPATH
        if not path.exists():
            raise ModalityNotFoundError(
                f"Tumour segmentation missing for {session.session_id}: {path}"
            )
        return load_nii(path)

    def load_brain_mask(self, session: LUMIERESession) -> NiftiVolume:
        """Load the DeepBraTumIA atlas-space binary brain mask."""
        path = session.root / _BRAIN_MASK_RELPATH
        if not path.exists():
            raise ModalityNotFoundError(f"Brain mask missing for {session.session_id}: {path}")
        return load_nii(path)

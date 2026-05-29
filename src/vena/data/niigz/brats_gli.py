"""BraTS-GLI pre-operative cohort loader for the NIfTI source release.

Source layout (one directory per session under ``source_root``):

    BraTS-GLI-PPPPP-TTT/
        BraTS-GLI-PPPPP-TTT-t1n.nii.gz
        BraTS-GLI-PPPPP-TTT-t1c.nii.gz
        BraTS-GLI-PPPPP-TTT-t2w.nii.gz
        BraTS-GLI-PPPPP-TTT-t2f.nii.gz
        BraTS-GLI-PPPPP-TTT-seg.nii.gz

All sessions share (182, 218, 182) voxels at 1 mm isotropic, skull-stripped,
axis codes LAS. Patient ID is derived by stripping the trailing timepoint
suffix (-TTT) from the session folder name. No metadata CSV exists.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SESSION_DIR_RE = re.compile(r"^BraTS-GLI-(\d+)-(\d+)$")


@dataclass(frozen=True)
class BraTSGLISession:
    """A single BraTS-GLI session handle.

    Attributes
    ----------
    session_id : str
        Full folder name, e.g. ``BraTS-GLI-00001-000``.
    patient_id : str
        Patient-level identifier with timepoint stripped, e.g. ``BraTS-GLI-00001``.
    root : Path
        Absolute path to the session directory.
    """

    session_id: str
    patient_id: str
    root: Path


class BraTSGLIDataset:
    """Index of the BraTS-GLI pre-operative source cohort.

    Parameters
    ----------
    source_root
        Directory containing ``BraTS-GLI-PPPPP-TTT/`` subdirectories.

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
            "BraTS-GLI: discovered %d sessions from %d patients",
            len(self._sessions),
            len(self._patient_groups),
        )

    def _discover(
        self,
    ) -> tuple[list[BraTSGLISession], list[tuple[str, list[int]]]]:
        """Discover sessions and build a CSR-ready patient grouping.

        Sessions are sorted first by patient ID then by session ID within each
        patient, guaranteeing that all sessions of a patient are contiguous.

        Returns
        -------
        sessions
            Ordered list of :class:`BraTSGLISession`.
        patient_groups
            ``(patient_id, [row_indices])`` pairs, one per patient, in the same
            order as ``sessions``.
        """
        # Collect (patient_id, session_id, path) triples.
        raw: list[tuple[str, str, Path]] = []
        for d in self.source_root.iterdir():
            if not d.is_dir():
                continue
            m = _SESSION_DIR_RE.match(d.name)
            if m is None:
                continue
            patient_id = f"BraTS-GLI-{m.group(1)}"
            raw.append((patient_id, d.name, d))

        # Sort: outer key = patient_id (lexicographic), inner = session_id.
        raw.sort(key=lambda t: (t[0], t[1]))

        sessions: list[BraTSGLISession] = []
        patient_groups: list[tuple[str, list[int]]] = []
        current_patient: str | None = None
        current_indices: list[int] = []

        for i, (patient_id, session_id, path) in enumerate(raw):
            sessions.append(BraTSGLISession(session_id=session_id, patient_id=patient_id, root=path))
            if patient_id != current_patient:
                if current_patient is not None:
                    patient_groups.append((current_patient, current_indices))
                current_patient = patient_id
                current_indices = [i]
            else:
                current_indices.append(i)

        if current_patient is not None:
            patient_groups.append((current_patient, current_indices))

        return sessions, patient_groups

    # ----- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._sessions)

    def __iter__(self) -> Iterator[BraTSGLISession]:
        return iter(self._sessions)

    def __getitem__(self, index: int) -> BraTSGLISession:
        return self._sessions[index]

    # ----- cohort-level accessors ---------------------------------------------

    def sessions(self) -> list[BraTSGLISession]:
        """Return all sessions in CSR-compatible order (patients contiguous)."""
        return list(self._sessions)

    def patient_groups(self) -> list[tuple[str, list[int]]]:
        """Return ``(patient_id, [row_indices])`` pairs in patient-sorted order."""
        return list(self._patient_groups)

    def patient_ids(self) -> list[str]:
        """Return unique patient IDs in sorted order."""
        return [pid for pid, _ in self._patient_groups]

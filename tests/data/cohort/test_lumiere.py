"""LUMIERE NIfTI reader (longitudinal, DeepBraTumIA atlas/skull-strip path).

Synthetic on-disk fixture. Covers registration, session discovery, CSR
patient grouping, modality / segmentation / brain-mask IO, missing-file
skip behaviour, and CohortProtocol conformance.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vena.data.cohort import CohortProtocol, get_cohort_registry
from vena.data.niigz import LUMIEREDataset, LUMIERESession

pytestmark = pytest.mark.unit


_SHAPE: tuple[int, int, int] = (8, 8, 8)
_LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def _write_nii(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), _LPS_AFFINE)
    nib.save(img, str(path))


def _build_session(session_dir: Path, *, complete: bool = True) -> None:
    rng = np.random.default_rng(0)
    skull = session_dir / "DeepBraTumIA-segmentation" / "atlas" / "skull_strip"
    seg = session_dir / "DeepBraTumIA-segmentation" / "atlas" / "segmentation"
    files = {
        "t1_skull_strip.nii.gz": rng.uniform(0, 200, _SHAPE).astype(np.float32),
        "ct1_skull_strip.nii.gz": rng.uniform(0, 200, _SHAPE).astype(np.float32),
        "t2_skull_strip.nii.gz": rng.uniform(0, 200, _SHAPE).astype(np.float32),
        "flair_skull_strip.nii.gz": rng.uniform(0, 200, _SHAPE).astype(np.float32),
    }
    if not complete:
        files.pop("ct1_skull_strip.nii.gz")
    for name, arr in files.items():
        _write_nii(skull / name, arr)
    brain = np.zeros(_SHAPE, dtype=np.float32)
    brain[2:6, 2:6, 2:6] = 1.0
    _write_nii(skull / "brain_mask.nii.gz", brain)
    seg_arr = np.zeros(_SHAPE, dtype=np.float32)
    seg_arr[3:5, 3:5, 3:5] = 3.0  # BraTS-2023 ET
    seg_arr[2:6, 2:6, 2:6][seg_arr[2:6, 2:6, 2:6] == 0] = 1.0
    _write_nii(seg / "seg_mask.nii.gz", seg_arr)


def _build_lumiere_tree(
    root: Path,
    *,
    patient_sessions: list[tuple[str, list[str]]],
    incomplete: set[tuple[str, str]] | None = None,
) -> None:
    incomplete = incomplete or set()
    for patient_id, session_names in patient_sessions:
        for sname in session_names:
            session_dir = root / patient_id / sname
            _build_session(session_dir, complete=(patient_id, sname) not in incomplete)


def test_lumiere_registered() -> None:
    reg = get_cohort_registry()
    assert "lumiere" in reg
    assert reg.pathology_of("lumiere") == "glioma"


def test_reader_discovers_sessions_with_csr_grouping(tmp_path: Path) -> None:
    _build_lumiere_tree(
        tmp_path,
        patient_sessions=[
            ("Patient-001", ["week-000-1", "week-000-2", "week-044", "week-056"]),
            ("Patient-002", ["week-000", "week-003", "week-021"]),
        ],
    )
    ds = LUMIEREDataset(tmp_path)
    assert isinstance(ds, CohortProtocol)
    assert len(ds) == 7
    groups = ds.patient_groups()
    assert [pid for pid, _ in groups] == ["Patient-001", "Patient-002"]
    # CSR contiguity per patient.
    p1_indices = groups[0][1]
    assert p1_indices == [0, 1, 2, 3]
    assert p1_indices == sorted(p1_indices)
    # Ordering within a patient: week ascending, repeat ascending (-1 first).
    p1_sessions = [ds[i] for i in p1_indices]
    assert [s.metadata["week"] for s in p1_sessions] == [0, 0, 44, 56]
    assert [s.metadata["week_repeat"] for s in p1_sessions] == [1, 2, -1, -1]


def test_reader_skips_incomplete_session(tmp_path: Path) -> None:
    _build_lumiere_tree(
        tmp_path,
        patient_sessions=[
            ("Patient-001", ["week-000-1", "week-000-2"]),
        ],
        incomplete={("Patient-001", "week-000-2")},
    )
    ds = LUMIEREDataset(tmp_path)
    assert len(ds) == 1
    assert ds[0].session_id == "Patient-001__week-000-1"


def test_reader_load_modality_and_seg(tmp_path: Path) -> None:
    _build_lumiere_tree(
        tmp_path,
        patient_sessions=[("Patient-001", ["week-000"])],
    )
    ds = LUMIEREDataset(tmp_path)
    s = ds[0]
    assert isinstance(s, LUMIERESession)
    for mod in ("t1pre", "t1c", "t2", "flair"):
        vol = ds.load_modality(s, mod)
        assert vol.array.shape == _SHAPE
    seg = ds.load_tumor_seg(s)
    assert set(np.unique(np.asarray(seg.array)).tolist()) <= {0.0, 1.0, 3.0}
    brain = ds.load_brain_mask(s)
    assert brain.array.shape == _SHAPE

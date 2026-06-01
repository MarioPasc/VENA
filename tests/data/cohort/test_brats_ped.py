"""BraTS-PED NIfTI reader (pediatric high-grade glioma).

Synthetic on-disk fixture only. Covers registration, BraTS-PED pattern
discovery, modality / segmentation IO, and CohortProtocol conformance.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vena.data.cohort import CohortProtocol, get_cohort_registry
from vena.data.niigz import BraTSPedDataset, BraTSPedPatient

pytestmark = pytest.mark.unit


_SHAPE: tuple[int, int, int] = (8, 8, 8)
_LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def _write_nii(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), _LPS_AFFINE)
    nib.save(img, str(path))


def _build_brats_ped_tree(root: Path, *, patient_ids: list[str]) -> None:
    rng = np.random.default_rng(0)
    for pid in patient_ids:
        d = root / pid
        for suffix in ("t1n", "t1c", "t2w", "t2f"):
            # Pediatric BraTS-PED stripped data: non-negative intensities,
            # background exactly zero.
            arr = np.abs(rng.standard_normal(_SHAPE).astype(np.float32))
            arr[0, 0, 0] = 0.0
            _write_nii(d / f"{pid}-{suffix}.nii.gz", arr)
        seg = np.zeros(_SHAPE, dtype=np.float32)
        seg[3:5, 3:5, 3:5] = 3.0  # BraTS-2023 ET
        seg[2:6, 2:6, 2:6][seg[2:6, 2:6, 2:6] == 0] = 1.0
        _write_nii(d / f"{pid}-seg.nii.gz", seg)


def test_brats_ped_registered() -> None:
    reg = get_cohort_registry()
    assert "brats_ped" in reg
    assert reg.pathology_of("brats_ped") == "glioma"


def test_brats_ped_reader_discovers_patients(tmp_path: Path) -> None:
    subset = tmp_path / "BraTS-PEDs2024_Training"
    _build_brats_ped_tree(
        subset,
        patient_ids=["BraTS-PED-00001-000", "BraTS-PED-00043-000", "BraTS-PED-00261-000"],
    )
    ds = BraTSPedDataset(subset)
    assert isinstance(ds, CohortProtocol)
    assert len(ds) == 3
    assert ds.ids() == [
        "BraTS-PED-00001-000",
        "BraTS-PED-00043-000",
        "BraTS-PED-00261-000",
    ]
    p = ds[0]
    assert isinstance(p, BraTSPedPatient)
    vol = ds.load_modality(p, "t1pre")
    assert vol.array.shape == _SHAPE
    seg = ds.load_tumor_seg(p)
    seg_unique = set(np.unique(np.asarray(seg.array)).tolist())
    assert seg_unique <= {0.0, 1.0, 3.0}


def test_brats_ped_lookup_by_id_and_index(tmp_path: Path) -> None:
    subset = tmp_path / "BraTS-PEDs2024_Training"
    _build_brats_ped_tree(subset, patient_ids=["BraTS-PED-00043-000"])
    ds = BraTSPedDataset(subset)
    assert ds[0].patient_id == "BraTS-PED-00043-000"
    assert ds["BraTS-PED-00043-000"].patient_id == "BraTS-PED-00043-000"


def test_brats_ped_rejects_non_brats_dir(tmp_path: Path) -> None:
    subset = tmp_path / "training"
    (subset / "garbage-dir").mkdir(parents=True)
    _build_brats_ped_tree(subset, patient_ids=["BraTS-PED-00043-000"])
    ds = BraTSPedDataset(subset)
    assert ds.ids() == ["BraTS-PED-00043-000"]

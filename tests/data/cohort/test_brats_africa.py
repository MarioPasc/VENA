"""BraTS-Africa NIfTI readers (glioma + other neoplasms).

Synthetic on-disk fixture only. Covers registration, BraTS-pattern discovery,
modality / segmentation IO, and CohortProtocol conformance for both subsets.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vena.data.cohort import CohortProtocol, get_cohort_registry
from vena.data.niigz import (
    BraTSAfricaGliomaDataset,
    BraTSAfricaOtherDataset,
    BraTSAfricaPatient,
)

pytestmark = pytest.mark.unit


_SHAPE: tuple[int, int, int] = (8, 8, 8)
_LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def _write_nii(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), _LPS_AFFINE)
    nib.save(img, str(path))


def _build_brats_africa_tree(
    root: Path,
    *,
    patient_ids: list[str],
) -> None:
    rng = np.random.default_rng(0)
    for pid in patient_ids:
        d = root / pid
        for suffix in ("t1n", "t1c", "t2w", "t2f"):
            # z-score-like data: include negative values inside the brain.
            arr = rng.standard_normal(_SHAPE).astype(np.float32)
            arr[0, 0, 0] = 0.0  # ensure at least one zero (background).
            _write_nii(d / f"{pid}-{suffix}.nii.gz", arr)
        seg = np.zeros(_SHAPE, dtype=np.float32)
        seg[3:5, 3:5, 3:5] = 3.0  # BraTS-2023 ET
        seg[2:6, 2:6, 2:6][seg[2:6, 2:6, 2:6] == 0] = 1.0
        _write_nii(d / f"{pid}-seg.nii.gz", seg)


def test_subsets_registered_with_distinct_pathology() -> None:
    reg = get_cohort_registry()
    assert "brats_africa_glioma" in reg
    assert "brats_africa_other" in reg
    assert reg.pathology_of("brats_africa_glioma") == "glioma"
    assert reg.pathology_of("brats_africa_other") == "other"


def test_glioma_reader_discovers_patients(tmp_path: Path) -> None:
    subset = tmp_path / "95_Glioma"
    _build_brats_africa_tree(
        subset,
        patient_ids=["BraTS-SSA-00002-000", "BraTS-SSA-00130-000", "BraTS-SSA-00230-000"],
    )
    ds = BraTSAfricaGliomaDataset(subset)
    assert isinstance(ds, CohortProtocol)
    assert len(ds) == 3
    assert ds.ids() == [
        "BraTS-SSA-00002-000",
        "BraTS-SSA-00130-000",
        "BraTS-SSA-00230-000",
    ]
    p = ds[0]
    assert isinstance(p, BraTSAfricaPatient)
    vol = ds.load_modality(p, "t1pre")
    assert vol.array.shape == _SHAPE
    seg = ds.load_tumor_seg(p)
    seg_unique = set(np.unique(np.asarray(seg.array)).tolist())
    assert seg_unique <= {0.0, 1.0, 3.0}


def test_other_reader_discovers_patients(tmp_path: Path) -> None:
    subset = tmp_path / "51_OtherNeoplasms"
    _build_brats_africa_tree(subset, patient_ids=["BraTS-SSA-00009-000", "BraTS-SSA-00170-000"])
    ds = BraTSAfricaOtherDataset(subset)
    assert isinstance(ds, CohortProtocol)
    assert ds.ids() == ["BraTS-SSA-00009-000", "BraTS-SSA-00170-000"]


def test_reader_lookup_by_id_and_index(tmp_path: Path) -> None:
    subset = tmp_path / "95_Glioma"
    _build_brats_africa_tree(subset, patient_ids=["BraTS-SSA-00002-000"])
    ds = BraTSAfricaGliomaDataset(subset)
    assert ds[0].patient_id == "BraTS-SSA-00002-000"
    assert ds["BraTS-SSA-00002-000"].patient_id == "BraTS-SSA-00002-000"

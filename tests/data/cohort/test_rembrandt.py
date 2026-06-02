"""REMBRANDT NIfTI reader.

Synthetic on-disk fixture only. Covers registration, the two ID families
(``900-00-*_date`` and ``HF*_date``), modality / GlistrBoost segmentation IO,
and CohortProtocol conformance.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vena.data.cohort import CohortProtocol, get_cohort_registry
from vena.data.niigz import REMBRANDTDataset, REMBRANDTPatient

pytestmark = pytest.mark.unit


_SHAPE: tuple[int, int, int] = (8, 8, 8)
_LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def _write_nii(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), _LPS_AFFINE)
    nib.save(img, str(path))


def _build_rembrandt_tree(root: Path, *, patient_ids: list[str]) -> None:
    """Write a 4-modality + GlistrBoost segmentation per patient.

    Modality filenames: ``<pid>_{t1,t1ce,t2,flair}_LPS_rSRI.nii.gz``.
    Seg filename:       ``<pid>_GlistrBoost_out.nii.gz`` (BraTS-2021 labels).
    """
    rng = np.random.default_rng(0)
    for pid in patient_ids:
        d = root / pid
        for suffix in ("t1", "t1ce", "t2", "flair"):
            arr = rng.standard_normal(_SHAPE).astype(np.float32)
            arr[0, 0, 0] = 0.0  # ensure at least one zero (background sentinel).
            _write_nii(d / f"{pid}_{suffix}_LPS_rSRI.nii.gz", arr)
        seg = np.zeros(_SHAPE, dtype=np.float32)
        seg[3:5, 3:5, 3:5] = 4.0  # BraTS-2021 ET
        seg[2:6, 2:6, 2:6][seg[2:6, 2:6, 2:6] == 0] = 1.0  # NCR/NET
        _write_nii(d / f"{pid}_GlistrBoost_out.nii.gz", seg)


def test_registered_with_glioma_pathology() -> None:
    reg = get_cohort_registry()
    assert "rembrandt" in reg
    assert reg.pathology_of("rembrandt") == "glioma"


def test_reader_discovers_both_id_families(tmp_path: Path) -> None:
    _build_rembrandt_tree(
        tmp_path,
        patient_ids=[
            "900-00-5303_2005.03.24",
            "900-00-5308_2005.04.24",
            "HF1318_1994.04.23",
            "HF1325_1994.05.07",
        ],
    )
    # An obviously bogus directory must be ignored by the regex.
    (tmp_path / "README_notes").mkdir()

    ds = REMBRANDTDataset(tmp_path)
    assert isinstance(ds, CohortProtocol)
    assert len(ds) == 4
    assert ds.ids() == sorted(
        [
            "900-00-5303_2005.03.24",
            "900-00-5308_2005.04.24",
            "HF1318_1994.04.23",
            "HF1325_1994.05.07",
        ]
    )


def test_modality_and_segmentation_loading(tmp_path: Path) -> None:
    _build_rembrandt_tree(tmp_path, patient_ids=["HF1318_1994.04.23"])
    ds = REMBRANDTDataset(tmp_path)
    p = ds[0]
    assert isinstance(p, REMBRANDTPatient)

    for name in ("t1pre", "t1c", "t2", "flair"):
        vol = ds.load_modality(p, name)
        assert vol.array.shape == _SHAPE

    seg = ds.load_tumor_seg(p)
    seg_unique = set(np.unique(np.asarray(seg.array)).tolist())
    assert seg_unique <= {0.0, 1.0, 4.0}  # BraTS-2021 labels subset (no ED in fixture)


def test_reader_lookup_by_id_and_index(tmp_path: Path) -> None:
    _build_rembrandt_tree(tmp_path, patient_ids=["900-00-5303_2005.03.24"])
    ds = REMBRANDTDataset(tmp_path)
    assert ds[0].patient_id == "900-00-5303_2005.03.24"
    assert ds["900-00-5303_2005.03.24"].patient_id == "900-00-5303_2005.03.24"


def test_bogus_directory_rejected(tmp_path: Path) -> None:
    _build_rembrandt_tree(tmp_path, patient_ids=["HF1318_1994.04.23"])
    # Same date but wrong subject prefix — must not be picked up.
    (tmp_path / "PatientFOO_1994.04.23").mkdir()
    # Right prefix, wrong date format.
    (tmp_path / "HF1318_1994-04-23").mkdir()
    ds = REMBRANDTDataset(tmp_path)
    assert ds.ids() == ["HF1318_1994.04.23"]

"""UPENN-GBM NIfTI reader (preoperative adult GBM).

Synthetic on-disk fixture only. Covers registration, UPENN-GBM directory
pattern discovery, modality + manual/automated seg fallback, BraTS-21
lookup metadata join, ``seg_source`` tagging, and CohortProtocol
conformance.
"""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vena.data.cohort import CohortProtocol, get_cohort_registry
from vena.data.niigz import UPENNGBMDataset, UPENNGBMPatient

pytestmark = pytest.mark.unit


_SHAPE: tuple[int, int, int] = (8, 8, 8)
_LPS_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def _write_nii(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), _LPS_AFFINE)
    nib.save(img, str(path))


def _write_modality_dir(images_structural: Path, pid: str) -> None:
    rng = np.random.default_rng(seed=hash(pid) % (2**32))
    pdir = images_structural / pid
    for suffix in ("T1", "T1GD", "T2", "FLAIR"):
        arr = np.abs(rng.standard_normal(_SHAPE).astype(np.float32))
        arr[0, 0, 0] = 0.0  # leave a background voxel
        _write_nii(pdir / f"{pid}_{suffix}.nii.gz", arr)


def _write_seg(path: Path, *, label_set: tuple[int, ...] = (1, 2, 4)) -> None:
    seg = np.zeros(_SHAPE, dtype=np.float32)
    seg[3:5, 3:5, 3:5] = float(label_set[0])
    if len(label_set) > 1:
        seg[5:6, 5:6, 5:6] = float(label_set[1])
    if len(label_set) > 2:
        seg[2:3, 2:3, 2:3] = float(label_set[2])
    _write_nii(path, seg)


def _build_upenn_tree(
    nifti_files_root: Path,
    *,
    patient_ids_with_manual: list[str],
    patient_ids_with_auto: list[str],
    patient_ids_no_seg: list[str],
) -> Path:
    """Return the ``images_structural`` source root expected by the reader."""
    images_structural = nifti_files_root / "images_structural"
    images_segm = nifti_files_root / "images_segm"
    automated_segm = nifti_files_root / "automated_segm"
    for pid in patient_ids_with_manual + patient_ids_with_auto + patient_ids_no_seg:
        _write_modality_dir(images_structural, pid)
    for pid in patient_ids_with_manual:
        _write_seg(images_segm / f"{pid}_segm.nii.gz")
    for pid in patient_ids_with_auto:
        _write_seg(automated_segm / f"{pid}_automated_approx_segm.nii.gz")
    return images_structural


def _write_lookup_csv(
    csv_path: Path,
    *,
    rows: list[tuple[str, str, str]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "brats21_id", "brats21_data_collection"])
        w.writeheader()
        for pid, b21, dc in rows:
            w.writerow({"patient_id": pid, "brats21_id": b21, "brats21_data_collection": dc})


def test_upenn_gbm_registered() -> None:
    reg = get_cohort_registry()
    assert "upenn_gbm" in reg
    assert reg.pathology_of("upenn_gbm") == "glioma"


def test_upenn_gbm_reader_discovers_patients(tmp_path: Path) -> None:
    nifti_files = tmp_path / "NIfTI-files"
    source_root = _build_upenn_tree(
        nifti_files,
        patient_ids_with_manual=["UPENN-GBM-00002_11"],
        patient_ids_with_auto=["UPENN-GBM-00001_11", "UPENN-GBM-00004_11"],
        patient_ids_no_seg=["UPENN-GBM-00003_11"],
    )
    ds = UPENNGBMDataset(source_root)
    assert isinstance(ds, CohortProtocol)
    # 3 patients with seg; the no-seg one is dropped at discovery.
    assert len(ds) == 3
    assert ds.ids() == [
        "UPENN-GBM-00001_11",
        "UPENN-GBM-00002_11",
        "UPENN-GBM-00004_11",
    ]
    p = ds[0]
    assert isinstance(p, UPENNGBMPatient)


def test_upenn_gbm_modality_loads(tmp_path: Path) -> None:
    nifti_files = tmp_path / "NIfTI-files"
    source_root = _build_upenn_tree(
        nifti_files,
        patient_ids_with_manual=["UPENN-GBM-00002_11"],
        patient_ids_with_auto=[],
        patient_ids_no_seg=[],
    )
    ds = UPENNGBMDataset(source_root)
    p = ds["UPENN-GBM-00002_11"]
    for slug in ("t1pre", "t1c", "t2", "flair"):
        vol = ds.load_modality(p, slug)
        assert vol.array.shape == _SHAPE
    seg = ds.load_tumor_seg(p)
    seg_unique = set(np.unique(np.asarray(seg.array)).astype(int).tolist())
    assert seg_unique <= {0, 1, 2, 4}


def test_upenn_gbm_prefers_manual_over_auto(tmp_path: Path) -> None:
    nifti_files = tmp_path / "NIfTI-files"
    pid = "UPENN-GBM-00002_11"
    source_root = _build_upenn_tree(
        nifti_files,
        patient_ids_with_manual=[pid],
        patient_ids_with_auto=[pid],  # also has an automated seg
        patient_ids_no_seg=[],
    )
    ds = UPENNGBMDataset(source_root)
    p = ds[pid]
    # Manual seg present → seg_source should be 'manual'.
    assert p.metadata.get("seg_source") == "manual"
    # And the resolved seg path should be the manual one.
    seg_path, source = ds._resolve_seg(pid)
    assert source == "manual"
    assert seg_path is not None
    assert seg_path.name.endswith("_segm.nii.gz")
    assert "automated" not in seg_path.name


def test_upenn_gbm_falls_back_to_auto(tmp_path: Path) -> None:
    nifti_files = tmp_path / "NIfTI-files"
    pid = "UPENN-GBM-00001_11"
    source_root = _build_upenn_tree(
        nifti_files,
        patient_ids_with_manual=[],
        patient_ids_with_auto=[pid],
        patient_ids_no_seg=[],
    )
    ds = UPENNGBMDataset(source_root)
    p = ds[pid]
    assert p.metadata.get("seg_source") == "automated"


def test_upenn_gbm_brats21_lookup_join(tmp_path: Path) -> None:
    nifti_files = tmp_path / "NIfTI-files"
    pid_mapped = "UPENN-GBM-00011_11"
    pid_unmapped = "UPENN-GBM-99999_11"
    source_root = _build_upenn_tree(
        nifti_files,
        patient_ids_with_manual=[pid_mapped],
        patient_ids_with_auto=[pid_unmapped],
        patient_ids_no_seg=[],
    )
    csv_path = tmp_path / "metadata" / "UPENN-GBM_brats21_lookup_v1.csv"
    _write_lookup_csv(
        csv_path,
        rows=[(pid_mapped, "BraTS2021_00131", "UPENN-GBM")],
    )
    ds = UPENNGBMDataset(source_root, metadata_csv=csv_path)
    mapped = ds[pid_mapped]
    assert mapped.metadata.get("brats21_id") == "BraTS2021_00131"
    assert mapped.metadata.get("brats21_data_collection") == "UPENN-GBM"
    unmapped = ds[pid_unmapped]
    # Unmapped patient: brats21 fields absent (empty bridge → dedup passes).
    assert "brats21_id" not in unmapped.metadata


def test_upenn_gbm_lookup_by_id_and_index(tmp_path: Path) -> None:
    nifti_files = tmp_path / "NIfTI-files"
    source_root = _build_upenn_tree(
        nifti_files,
        patient_ids_with_manual=[],
        patient_ids_with_auto=["UPENN-GBM-00043_11"],
        patient_ids_no_seg=[],
    )
    ds = UPENNGBMDataset(source_root)
    assert ds[0].patient_id == "UPENN-GBM-00043_11"
    assert ds["UPENN-GBM-00043_11"].patient_id == "UPENN-GBM-00043_11"


def test_upenn_gbm_rejects_non_matching_dir(tmp_path: Path) -> None:
    nifti_files = tmp_path / "NIfTI-files"
    source_root = _build_upenn_tree(
        nifti_files,
        patient_ids_with_manual=[],
        patient_ids_with_auto=["UPENN-GBM-00043_11"],
        patient_ids_no_seg=[],
    )
    (source_root / "garbage-dir").mkdir(parents=True)
    (source_root / "UPENN-GBM-malformed").mkdir(parents=True)
    ds = UPENNGBMDataset(source_root)
    assert ds.ids() == ["UPENN-GBM-00043_11"]

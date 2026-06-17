"""Tests for image-domain helpers — focuses on CSR patient → scan expansion."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.registry import CohortEntry
from vena.inference.image_dataset import (
    ImageH5LookupError,
    resolve_test_patient_ids,
    row_index_for_patient,
)

pytestmark = pytest.mark.unit


def _vlen(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _write_longitudinal_image_h5(path: Path, role: str = "cv") -> None:
    """Write a 2-patient × 3-scan longitudinal cohort H5.

    Patient P001 has 1 scan, patient P002 has 2 scans (longitudinal).
    splits/test contains both patient IDs (P001, P002). Scan IDs in /ids
    are S001, S002a, S002b.
    """
    shape = (8, 8, 8)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0.0"
        f.attrs["cohort"] = "LONG"
        f.attrs["crop_box"] = json.dumps(list(shape))
        # 3 scans
        f.create_dataset(
            "ids", data=_vlen(["S001", "S002a", "S002b"]), dtype=h5py.string_dtype("utf-8")
        )
        for mod in ("t1pre", "t1c", "t2", "flair"):
            data = np.zeros((3, *shape), dtype=np.float32)
            f.create_dataset(
                f"images/{mod}",
                data=data,
                chunks=(1, *shape),
                compression="gzip",
                compression_opts=4,
            )
        f.create_dataset("masks/brain", data=np.ones((3, *shape), dtype=np.int8))
        f.create_dataset("masks/tumor", data=np.zeros((3, *shape), dtype=np.int8))
        f.create_dataset("crop/origin", data=np.zeros((3, 3), dtype=np.int32))
        # CSR: 2 patients, offsets [0, 1, 3]
        f.create_dataset(
            "patients/keys", data=_vlen(["P001", "P002"]), dtype=h5py.string_dtype("utf-8")
        )
        f.create_dataset("patients/offsets", data=np.asarray([0, 1, 3], dtype=np.int32))
        # splits/test contains PATIENT ids, not scan ids
        if role == "cv":
            f.create_dataset(
                "splits/test", data=_vlen(["P001", "P002"]), dtype=h5py.string_dtype("utf-8")
            )


def _make_cohort(name: str, image_h5: Path, latent_h5: Path, role: str = "cv") -> CohortEntry:
    return CohortEntry(
        name=name,
        pathology="glioma",
        label_system="BraTS2021",
        role=role,
        longitudinal=True,
        image_h5=image_h5,
        latent_h5=latent_h5,
        n_patients=2,
        n_scans=3,
        modalities=["t1pre", "t1c", "t2", "flair"],
        has_swan=False,
    )


def test_longitudinal_csr_expansion(tmp_path: Path) -> None:
    """splits/test patient_ids must expand to all their scan_ids."""
    image_h5 = tmp_path / "long_image.h5"
    latent_h5 = tmp_path / "long_latent.h5"
    _write_longitudinal_image_h5(image_h5, role="cv")
    latent_h5.write_bytes(b"\x00")
    cohort = _make_cohort("LONG", image_h5, latent_h5, role="cv")

    scan_ids = resolve_test_patient_ids(cohort, fold=0)
    # P001 has 1 scan (S001), P002 has 2 scans (S002a, S002b).
    assert scan_ids == ["S001", "S002a", "S002b"]

    # Every returned scan id must be lookup-able in /ids.
    for sid in scan_ids:
        assert row_index_for_patient(image_h5, sid) >= 0


def test_test_only_role_csr_expansion(tmp_path: Path) -> None:
    """role=test_only cohort: patients/keys (patient_ids) → expand to scans."""
    image_h5 = tmp_path / "long_image.h5"
    latent_h5 = tmp_path / "long_latent.h5"
    _write_longitudinal_image_h5(image_h5, role="test_only")
    latent_h5.write_bytes(b"\x00")
    cohort = _make_cohort("LONG", image_h5, latent_h5, role="test_only")

    scan_ids = resolve_test_patient_ids(cohort, fold=0)
    assert scan_ids == ["S001", "S002a", "S002b"]


def test_non_longitudinal_passthrough(tmp_path: Path, synthetic_cohort) -> None:
    """Non-longitudinal cohort: scan_id == patient_id, identity expansion."""
    cohort, image_h5 = synthetic_cohort
    scan_ids = resolve_test_patient_ids(cohort, fold=0)
    assert scan_ids == ["P001"]


def test_missing_image_h5_raises(tmp_path: Path) -> None:
    cohort = _make_cohort("X", tmp_path / "nope.h5", tmp_path / "nope2.h5")
    with pytest.raises(ImageH5LookupError):
        resolve_test_patient_ids(cohort, fold=0)

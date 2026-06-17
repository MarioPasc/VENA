"""Tests for the predictions-H5 writer + validator."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.inference.h5_writer import (
    SCHEMA_VERSION,
    PerPatientRecord,
    PredictionsH5Error,
    assert_predictions_valid,
    validate_predictions,
    write_predictions_h5,
)

pytestmark = pytest.mark.unit


def _make_record(pid: str, shape=(8, 8, 8), seed: int = 0) -> PerPatientRecord:
    rng = np.random.default_rng(seed)
    brain = np.zeros(shape, dtype=np.int8)
    brain[2:6, 2:6, 2:6] = 1
    real = rng.uniform(0.0, 1.0, size=shape).astype(np.float32) * brain
    synth = (real * 0.9).astype(np.float32)  # close to real, within brain mask
    raw = synth.copy()
    wt = np.zeros(shape, dtype=np.int8)
    wt[3:5, 3:5, 3:5] = 1
    return PerPatientRecord(
        patient_id=pid,
        cohort="TEST",
        t1c_synthetic_harmonised=synth,
        t1c_synthetic_raw=raw,
        t1c_real_harmonised=real,
        t1pre_harmonised=real.copy(),
        t2_harmonised=real.copy(),
        flair_harmonised=real.copy(),
        brain_mask=brain,
        wt_mask=wt,
        inference_seconds=0.1,
        peak_vram_mb=0.0,
    )


def test_write_and_validate_ok(tmp_path: Path) -> None:
    records = [_make_record(f"P{i:03d}", seed=i) for i in range(3)]
    path = tmp_path / "preds.h5"
    write_predictions_h5(path, records, method="C0-Identity", cohort="TEST", nfe=1, ring="A")
    assert_predictions_valid(path)  # no exception
    assert validate_predictions(path) == []


def test_root_attrs_present(tmp_path: Path) -> None:
    records = [_make_record("P001")]
    path = tmp_path / "preds.h5"
    write_predictions_h5(
        path,
        records,
        method="X",
        cohort="C",
        nfe=5,
        ring="A",
        git_sha="abc123",
        checkpoint_sha256="deadbeef",
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["schema_version"] == SCHEMA_VERSION
        assert f.attrs["method"] == "X"
        assert f.attrs["cohort"] == "C"
        assert int(f.attrs["nfe"]) == 5
        assert f.attrs["ring"] == "A"
        assert "harmonisation_recipe" in f.attrs
        assert f.attrs["git_sha"] == "abc123"
        assert f.attrs["checkpoint_sha256"] == "deadbeef"


def test_validator_rejects_nan(tmp_path: Path) -> None:
    r = _make_record("P001")
    r.t1c_synthetic_harmonised[0, 0, 0] = np.nan
    path = tmp_path / "bad.h5"
    write_predictions_h5(path, [r], method="X", cohort="C", nfe=1, ring="A")
    violations = validate_predictions(path)
    assert any("NaN" in v for v in violations)


def test_validator_rejects_out_of_range(tmp_path: Path) -> None:
    r = _make_record("P001")
    r.t1c_synthetic_harmonised[3, 3, 3] = 5.0  # out of [0, 1]
    path = tmp_path / "bad.h5"
    write_predictions_h5(path, [r], method="X", cohort="C", nfe=1, ring="A")
    violations = validate_predictions(path)
    assert any("out of [0, 1]" in v for v in violations)


def test_validator_rejects_nonzero_outside_brain(tmp_path: Path) -> None:
    r = _make_record("P001")
    # Force a nonzero value outside the brain mask.
    outside = r.brain_mask == 0
    r.t1c_synthetic_harmonised[outside] = 0.5
    path = tmp_path / "bad.h5"
    write_predictions_h5(path, [r], method="X", cohort="C", nfe=1, ring="A")
    violations = validate_predictions(path)
    assert any("nonzero outside brain" in v for v in violations)


def test_empty_records_raises(tmp_path: Path) -> None:
    with pytest.raises(PredictionsH5Error):
        write_predictions_h5(tmp_path / "x.h5", [], method="X", cohort="C", nfe=1, ring="A")


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    r1 = _make_record("P001", shape=(8, 8, 8))
    r2 = _make_record("P002", shape=(6, 6, 6))
    with pytest.raises(PredictionsH5Error):
        write_predictions_h5(tmp_path / "x.h5", [r1, r2], method="X", cohort="C", nfe=1, ring="A")


def test_duplicate_patient_id_caught_by_validator(tmp_path: Path) -> None:
    r1 = _make_record("DUP", seed=0)
    r2 = _make_record("DUP", seed=1)
    path = tmp_path / "dup.h5"
    write_predictions_h5(path, [r1, r2], method="X", cohort="C", nfe=1, ring="A")
    violations = validate_predictions(path)
    assert any("duplicates" in v for v in violations)

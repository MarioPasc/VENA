"""Schema 2.0: references live once per cohort, not in every prediction file.

Schema 1.1 copied the four reference modalities + a residual into every
prediction H5. They are invariant across NFE and method, so the benchmark
re-serialised them 45 times (once per method x NFE pair) — a record cost 24 MB
and the sweep projected to ~424 GB, over the home quota, ~70% duplicated bytes.

These tests pin the split: predictions carry only what varies, references are
written once and joinable, and no information is lost (the residual is exactly
real - synth, recomputable).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.inference.h5_writer import (
    PerPatientRecord,
    ReferencesH5Error,
    assert_references_valid,
    validate_predictions,
    validate_references,
    write_predictions_h5,
    write_references_h5,
)

pytestmark = pytest.mark.unit

_SHAPE = (6, 6, 4)


def _record(scan_id: str, *, patient_id: str | None = None, seed: int = 0) -> PerPatientRecord:
    rng = np.random.default_rng(seed)
    brain = np.ones(_SHAPE, dtype=np.int8)
    vol = rng.uniform(0.0, 1.0, size=_SHAPE).astype(np.float32)
    return PerPatientRecord(
        patient_id=patient_id or scan_id,
        scan_id=scan_id,
        cohort="TEST",
        t1c_synthetic_harmonised=vol,
        t1c_synthetic_raw=vol * 2.0,
        t1c_real_harmonised=rng.uniform(0.0, 1.0, size=_SHAPE).astype(np.float32),
        t1pre_harmonised=rng.uniform(0.0, 1.0, size=_SHAPE).astype(np.float32),
        t2_harmonised=rng.uniform(0.0, 1.0, size=_SHAPE).astype(np.float32),
        flair_harmonised=rng.uniform(0.0, 1.0, size=_SHAPE).astype(np.float32),
        brain_mask=brain,
        wt_mask=np.zeros(_SHAPE, dtype=np.int8),
        inference_seconds=1.0,
        peak_vram_mb=2.0,
    )


def _write_preds(path: Path, records: list[PerPatientRecord]) -> Path:
    return write_predictions_h5(
        path,
        records,
        method="M",
        cohort="TEST",
        nfe=5,
        ring="A",
        references_h5="references/TEST.h5",
    )


def test_predictions_no_longer_carry_references_or_residual(tmp_path: Path) -> None:
    """The duplicated groups must be GONE — that is the entire point of 2.0."""
    path = _write_preds(tmp_path / "p.h5", [_record("S1")])

    with h5py.File(path, "r") as f:
        assert "reference" not in f
        assert "residuals" not in f
        # what must remain: the varying data, plus masks so the file self-validates
        assert "predictions/t1c_synthetic_harmonised" in f
        assert "predictions/t1c_synthetic_raw" in f
        assert "masks/brain" in f
        assert f.attrs["schema_version"] == "2.0"
        assert f.attrs["references_h5"] == "references/TEST.h5"

    assert validate_predictions(path) == []


def test_prediction_without_reference_pointer_is_invalid(tmp_path: Path) -> None:
    """A prediction file nobody can score is a broken file, not a valid one."""
    path = write_predictions_h5(
        tmp_path / "p.h5", [_record("S1")], method="M", cohort="TEST", nfe=5, ring="A"
    )
    assert "missing root attr: references_h5" in validate_predictions(path)


def test_references_roundtrip_and_validate(tmp_path: Path) -> None:
    records = [_record(f"S{i}", seed=i) for i in range(3)]
    path = write_references_h5(tmp_path / "TEST.h5", records, cohort="TEST")

    assert_references_valid(path)
    assert validate_references(path) == []

    with h5py.File(path, "r") as f:
        assert f.attrs["schema_version"] == "2.0"
        assert f["reference/t1c_real_harmonised"].shape == (3, *_SHAPE)
        np.testing.assert_allclose(
            f["reference/t1c_real_harmonised"][1], records[1].t1c_real_harmonised
        )
        assert [s.decode() for s in f["metadata/scan_id"][:]] == ["S0", "S1", "S2"]


def test_residual_is_recoverable_by_joining_on_scan_id(tmp_path: Path) -> None:
    """Dropping residuals/raw loses nothing: real - synth reconstructs it exactly.

    Row order is deliberately NOT aligned between the two files here — the
    references file lists every scan of the cohort while the prediction file may
    have dropped a failed patient — so the join must be on scan_id, not index.
    """
    all_records = [_record(f"S{i}", seed=i) for i in range(3)]
    ref_path = write_references_h5(tmp_path / "TEST.h5", all_records, cohort="TEST")

    # prediction file holds only S2 and S0, in that order
    preds = [all_records[2], all_records[0]]
    pred_path = _write_preds(tmp_path / "p.h5", preds)

    with h5py.File(pred_path, "r") as pf, h5py.File(ref_path, "r") as rf:
        pred_scans = [s.decode() for s in pf["metadata/scan_id"][:]]
        ref_scans = [s.decode() for s in rf["metadata/scan_id"][:]]
        assert pred_scans == ["S2", "S0"] and ref_scans == ["S0", "S1", "S2"]

        for i, scan in enumerate(pred_scans):
            j = ref_scans.index(scan)
            synth = pf["predictions/t1c_synthetic_harmonised"][i]
            real = rf["reference/t1c_real_harmonised"][j]
            expected = all_records[2 if scan == "S2" else 0]
            np.testing.assert_allclose(synth, expected.t1c_synthetic_harmonised)
            np.testing.assert_allclose(real, expected.t1c_real_harmonised)
            # the residual schema 1.1 used to store, reconstructed exactly
            np.testing.assert_allclose(
                real - synth, expected.t1c_real_harmonised - expected.t1c_synthetic_harmonised
            )


def test_references_rejects_duplicate_scan_id(tmp_path: Path) -> None:
    """scan_id is the join key — duplicates would make the join ambiguous."""
    dupes = [_record("SAME", seed=0), _record("SAME", seed=1)]
    with pytest.raises(ReferencesH5Error, match="duplicate scan_id"):
        write_references_h5(tmp_path / "TEST.h5", dupes, cohort="TEST")


def test_references_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ReferencesH5Error, match="no records"):
        write_references_h5(tmp_path / "TEST.h5", [], cohort="TEST")


def test_prediction_payload_drops_the_duplicated_volumes(tmp_path: Path) -> None:
    """The saving is the reason this schema exists, so assert it actually lands.

    Schema 1.1 serialised 7 float32 volumes per record (2 synthetic + 4 reference
    + 1 residual); 2.0 serialises 2. Compare stored dataset bytes rather than file
    size — at test-volume scale HDF5's own metadata dwarfs the payload and would
    mask the effect.
    """
    records = [_record(f"S{i}", seed=i) for i in range(4)]
    pred = _write_preds(tmp_path / "p.h5", records)
    refs = write_references_h5(tmp_path / "TEST.h5", records, cohort="TEST")

    def float_payload(path: Path) -> int:
        total = 0

        def visit(_name: str, obj: object) -> None:
            nonlocal total
            if isinstance(obj, h5py.Dataset) and obj.dtype == np.float32 and obj.ndim == 4:
                total += obj.id.get_storage_size()

        with h5py.File(path, "r") as f:
            f.visititems(visit)
        return total

    pred_bytes = float_payload(pred)
    ref_bytes = float_payload(refs)

    # 2 float32 volumes/record here vs 7 under 1.1 -> the per-record prediction
    # payload is ~2/7 of what it was, and the 4 reference volumes are paid ONCE
    # per cohort instead of once per (method, NFE).
    assert 0 < pred_bytes < ref_bytes
    schema_1_1_payload = pred_bytes + ref_bytes + pred_bytes / 2  # +1 residual volume
    assert pred_bytes < 0.45 * schema_1_1_payload

"""Tests for vena.validation.audit.

Uses the conftest synth_shard fixture to avoid duplicating H5 creation logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.validation


def test_audit_harmonisation_columns(pred_path: Path) -> None:
    """audit_harmonisation returns the required column set."""
    from vena.validation.audit import audit_harmonisation
    from vena.validation.io import ReferenceCache, iter_scans

    cache = ReferenceCache()
    samples = list(iter_scans(pred_path, reference_cache=cache))
    df = audit_harmonisation(samples)

    required = {
        "scan_id",
        "patient_id",
        "cohort",
        "method",
        "nfe",
        "pred_in_range",
        "real_in_range",
        "pred_max_exterior",
        "real_max_exterior",
        "pred_min_brain",
        "pred_max_brain",
        "real_min_brain",
        "real_max_brain",
    }
    assert required.issubset(set(df.columns))


def test_audit_harmonisation_row_count(pred_path: Path) -> None:
    """One row per scan sample."""
    from vena.validation.audit import audit_harmonisation
    from vena.validation.io import ReferenceCache, iter_scans

    cache = ReferenceCache()
    samples = list(iter_scans(pred_path, reference_cache=cache))
    df = audit_harmonisation(samples)

    assert len(df) == len(samples)


def test_audit_harmonisation_in_range_for_valid_data(pred_path: Path) -> None:
    """Synthetic H5s in [0,1] — pred_in_range and real_in_range must be True."""
    from vena.validation.audit import audit_harmonisation
    from vena.validation.io import ReferenceCache, iter_scans

    cache = ReferenceCache()
    samples = list(iter_scans(pred_path, reference_cache=cache))
    df = audit_harmonisation(samples)

    assert df["pred_in_range"].all(), "All synthetic volumes should be in [0,1]"
    assert df["real_in_range"].all(), "All reference volumes should be in [0,1]"


def test_audit_harmonisation_detects_out_of_range(tmp_path: Path) -> None:
    """pred_in_range is False when a volume exceeds [0, 1] inside the brain."""

    from vena.validation.audit import audit_harmonisation
    from vena.validation.io import ScanSample

    h, w, d = 8, 8, 8
    brain = np.ones((h, w, d), dtype=bool)
    # Prediction has a voxel > 1.0 inside the brain.
    pred = np.full((h, w, d), 1.5, dtype=np.float32)
    real = np.zeros((h, w, d), dtype=np.float32)
    wt = np.zeros((h, w, d), dtype=bool)

    sample = ScanSample(
        scan_id="bad_scan",
        patient_id="pt1",
        cohort="Test",
        ring="A",
        method="VENA-test",
        nfe=5,
        pred=pred,
        pred_raw=pred,
        pred_harmonised=pred,
        pred_mode="raw",
        raw_p995=float(np.percentile(pred[brain], 99.5)),
        real=real,
        brain=brain,
        wt=wt,
        inference_seconds=1.0,
        peak_vram_mb=1000.0,
    )
    df = audit_harmonisation([sample])
    assert len(df) == 1
    assert not df.iloc[0]["pred_in_range"]
    assert df.iloc[0]["real_in_range"]  # real is all-zero, valid


def test_audit_harmonisation_empty_input() -> None:
    """Empty iterable returns a DataFrame with the correct columns and 0 rows."""
    from vena.validation.audit import audit_harmonisation

    df = audit_harmonisation([])
    assert len(df) == 0
    assert "scan_id" in df.columns
    assert "pred_in_range" in df.columns

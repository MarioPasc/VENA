"""Unit tests for the exhaustive-val box-comparison helpers.

Covers:
- ``build_crop_spec_from_h5``: constructs a CropPadSpec from a synthetic image H5.
- ``load_real_t1c_box``: crops and normalises a native volume to the box shape.
- ``full_volume_psnr_ssim``: computes PSNR/SSIM without error on box tensors.
- ``cohort`` column appears as the first column in the metrics CSV written by
  ``ExhaustiveValEngine._write_metrics_csv``.
- ``foreground_only=True`` default on ``load_real_t1c_normalised``.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vena.model.autoencoder.maisi.preprocessing import CropPadSpec
from vena.model.fm.eval.exhaustive import (
    build_crop_spec_from_h5,
    load_real_t1c_box,
    load_real_t1c_normalised,
    full_volume_psnr_ssim,
)
from vena.model.fm.metrics import ImageMetrics


# ---------------------------------------------------------------------------
# Synthetic H5 fixtures
# ---------------------------------------------------------------------------


def _make_image_h5(tmp_path: Path, native_shape=(12, 14, 12), target_shape=(8, 8, 8)) -> Path:
    """Write a minimal image H5 with schema-2.0.0 fields."""
    p = tmp_path / "image.h5"
    n = 2  # two patients
    rng = np.random.default_rng(0)
    images = rng.random((n, *native_shape), dtype=np.float32) * 1000.0
    # crop/origin: start the box at voxel (2, 3, 2) for both patients
    crop_origin = np.array([[2, 3, 2], [2, 3, 2]], dtype=np.int32)
    with h5py.File(p, "w") as f:
        f.create_dataset("ids", data=np.array([b"PID-A", b"PID-B"]))
        f.create_dataset("images/t1c", data=images)
        f.create_dataset("crop/origin", data=crop_origin)
        f.attrs["crop_box"] = json.dumps(list(target_shape))
    return p


# ---------------------------------------------------------------------------
# build_crop_spec_from_h5
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_crop_spec_returns_correct_fields(tmp_path: Path) -> None:
    native = (12, 14, 12)
    target = (8, 8, 8)
    h5 = _make_image_h5(tmp_path, native_shape=native, target_shape=target)
    spec = build_crop_spec_from_h5(h5, "PID-A")
    assert isinstance(spec, CropPadSpec)
    assert spec.crop_origin == (2, 3, 2)
    assert spec.native_shape == native
    assert spec.target_shape == target


@pytest.mark.unit
def test_build_crop_spec_raises_for_missing_patient(tmp_path: Path) -> None:
    from vena.model.fm.eval.exhaustive import ExhaustiveValError
    h5 = _make_image_h5(tmp_path)
    with pytest.raises(ExhaustiveValError, match="not found"):
        build_crop_spec_from_h5(h5, "NONEXISTENT")


@pytest.mark.unit
def test_build_crop_spec_raises_for_missing_crop_origin(tmp_path: Path) -> None:
    """H5 without crop/origin triggers ExhaustiveValError."""
    from vena.model.fm.eval.exhaustive import ExhaustiveValError
    p = tmp_path / "no_crop.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("ids", data=np.array([b"PID-A"]))
        f.create_dataset("images/t1c", data=np.zeros((1, 8, 8, 8), dtype=np.float32))
        f.attrs["crop_box"] = json.dumps([4, 4, 4])
        # Intentionally omit crop/origin
    with pytest.raises(ExhaustiveValError, match="crop/origin"):
        build_crop_spec_from_h5(p, "PID-A")


# ---------------------------------------------------------------------------
# load_real_t1c_box
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_real_t1c_box_output_shape(tmp_path: Path) -> None:
    target = (8, 8, 8)
    h5 = _make_image_h5(tmp_path, native_shape=(12, 14, 12), target_shape=target)
    spec = build_crop_spec_from_h5(h5, "PID-A")
    out = load_real_t1c_box(h5, "PID-A", spec)
    assert out.shape == target, f"expected {target}, got {tuple(out.shape)}"


@pytest.mark.unit
def test_load_real_t1c_box_range(tmp_path: Path) -> None:
    target = (8, 8, 8)
    h5 = _make_image_h5(tmp_path, native_shape=(12, 14, 12), target_shape=target)
    spec = build_crop_spec_from_h5(h5, "PID-A")
    out = load_real_t1c_box(h5, "PID-A", spec)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.unit
def test_load_real_t1c_box_deterministic(tmp_path: Path) -> None:
    """Same patient loaded twice must be identical."""
    h5 = _make_image_h5(tmp_path)
    spec = build_crop_spec_from_h5(h5, "PID-A")
    a = load_real_t1c_box(h5, "PID-A", spec)
    b = load_real_t1c_box(h5, "PID-A", spec)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# full_volume_psnr_ssim on box tensors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_psnr_ssim_box_no_error() -> None:
    """PSNR/SSIM compute without error on synthetic box volumes.

    SSIM is not bounded to [0,1] for random uncorrelated inputs on small
    volumes — we only assert the types are correct and values are finite.
    """
    metrics = ImageMetrics(data_range=1.0)
    pred = torch.rand(8, 8, 8)
    real = torch.rand(8, 8, 8)
    psnr, ssim = full_volume_psnr_ssim(pred, real, metrics)
    assert isinstance(psnr, float)
    assert isinstance(ssim, float)
    import math
    assert math.isfinite(psnr)
    assert math.isfinite(ssim)


@pytest.mark.unit
def test_psnr_ssim_identical_volumes() -> None:
    """Perfect prediction → PSNR very high, SSIM ≈ 1."""
    metrics = ImageMetrics(data_range=1.0)
    vol = torch.rand(8, 8, 8)
    psnr, ssim = full_volume_psnr_ssim(vol, vol, metrics)
    assert psnr > 60.0, f"expected high PSNR for identical volumes, got {psnr}"
    assert ssim == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# metrics CSV cohort column
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_csv_cohort_first_column(tmp_path: Path) -> None:
    """cohort is the first column and is written correctly."""
    from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

    rows = [
        {
            "cohort": "UCSF-PDGM",
            "patient_id": "PID-A",
            "nfe": 1,
            "psnr_db": 30.1,
            "ssim": 0.85,
            "latent_mse": 0.01,
            "latent_l1": 0.05,
            "latent_cosine": 0.99,
            "gen_sec": 0.5,
            "decode_sec": 0.2,
        },
        {
            "cohort": "BraTS-GLI",
            "patient_id": "PID-B",
            "nfe": 5,
            "psnr_db": 28.0,
            "ssim": 0.80,
            "latent_mse": 0.02,
            "latent_l1": 0.06,
            "latent_cosine": 0.98,
            "gen_sec": 1.0,
            "decode_sec": 0.3,
        },
    ]
    csv_path = tmp_path / "metrics.csv"
    ExhaustiveValEngine._write_metrics_csv(csv_path, rows)

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        assert fieldnames[0] == "cohort", f"first column should be 'cohort', got {fieldnames[0]}"
        data_rows = list(reader)

    assert data_rows[0]["cohort"] == "UCSF-PDGM"
    assert data_rows[1]["cohort"] == "BraTS-GLI"
    assert data_rows[0]["patient_id"] == "PID-A"


# ---------------------------------------------------------------------------
# load_real_t1c_normalised default foreground_only
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_real_t1c_normalised_default_foreground_only(tmp_path: Path) -> None:
    """Default foreground_only=True (skull-stripped brain standard)."""
    import inspect
    from vena.model.fm.eval.exhaustive import load_real_t1c_normalised as fn
    sig = inspect.signature(fn)
    default = sig.parameters["foreground_only"].default
    assert default is True, (
        f"load_real_t1c_normalised foreground_only default should be True, got {default!r}"
    )

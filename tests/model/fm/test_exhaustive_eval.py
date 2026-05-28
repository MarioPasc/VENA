"""Unit tests for the exhaustive image-space validation library helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vena.model.fm.eval.exhaustive import (
    load_real_t1c_normalised,
    select_content_slices,
    write_latent_preds_h5,
)


@pytest.mark.unit
def test_select_content_slices_offsets_and_equispaces() -> None:
    # Volume (H, W, D); content only on axial slices [20, 80).
    vol = torch.zeros(4, 4, 100)
    vol[..., 20:80] = 1.0
    idx = select_content_slices(vol, n_slices=10, offset=10)
    assert len(idx) == 10
    # content [20, 79]; offset inward by 10 -> [30, 69]
    assert idx[0] == 30
    assert idx[-1] == 69
    assert idx == sorted(idx)


@pytest.mark.unit
def test_select_content_slices_degenerate_falls_back() -> None:
    vol = torch.zeros(2, 2, 12)
    vol[..., 5] = 1.0  # single content slice; offset would collapse
    idx = select_content_slices(vol, n_slices=5, offset=10)
    assert len(idx) == 5
    assert all(0 <= k < 12 for k in idx)


@pytest.mark.unit
def test_load_real_t1c_normalised_matches_encoder_range(tmp_path: Path) -> None:
    p = tmp_path / "img.h5"
    raw = np.zeros((2, 8, 8, 6), dtype=np.float32)
    raw[0, ..., :] = np.linspace(0.0, 8000.0, 8 * 8 * 6).reshape(8, 8, 6)
    with h5py.File(p, "w") as f:
        f.create_dataset("ids", data=np.array([b"PID-A", b"PID-B"]))
        f.create_dataset("images/t1c", data=raw)
    out = load_real_t1c_normalised(p, "PID-A")
    assert out.shape == (8, 8, 6)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert float(out.max()) == pytest.approx(1.0, abs=1e-4)  # percentile-clipped to [0,1]


@pytest.mark.unit
def test_write_latent_preds_h5_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "preds.h5"
    entries = [
        ("PID-A", 1, np.random.rand(4, 6, 6, 4).astype(np.float32)),
        ("PID-A", 10, np.random.rand(4, 6, 6, 4).astype(np.float32)),
        ("PID-B", 1, np.random.rand(4, 6, 6, 4).astype(np.float32)),
    ]
    write_latent_preds_h5(p, entries, epoch=3, run_id="run-x")
    with h5py.File(p, "r") as f:
        assert f.attrs["schema_version"] == "1.0"
        assert int(f.attrs["epoch"]) == 3
        assert set(f["predictions"].keys()) == {"PID-A", "PID-B"}
        assert set(f["predictions/PID-A"].keys()) == {"nfe_1", "nfe_10"}
        assert tuple(f["predictions/PID-A/nfe_1"].shape) == (4, 6, 6, 4)

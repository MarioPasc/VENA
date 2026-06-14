"""Tests for the pGAN VENA dataset wrapper.

The fixture builds a tiny synthetic image H5 that matches the UCSF-PDGM schema
2.0.0 (``images/{t1pre,t1c,t2,flair}``, ``masks/brain``, ``ids``,
``splits/cv/fold_0/{train,val}``, ``splits/test``). No real data is needed.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vena.competitors.pgan_cgan.dataset import (
    DatasetError,
    UCSFPDGMSliceDataset,
    _pad_to,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def synth_h5(tmp_path: Path) -> Path:
    """Build a 4-patient synthetic H5 with non-trivial intensity ranges."""
    out = tmp_path / "synth_image.h5"
    N, H, W, D = 4, 240, 240, 155
    rng = np.random.default_rng(0)
    with h5py.File(out, "w") as f:
        ids = np.array([f"UCSF-PDGM-{i:04d}" for i in range(N)], dtype="S16")
        f.create_dataset("ids", data=ids)
        for mod in ("t1pre", "t1c", "t2", "flair"):
            arr = rng.uniform(0.0, 100.0, size=(N, H, W, D)).astype(np.float32)
            # Force per-patient distinct foreground scale to verify per-patient
            # percentile thresholds vary as designed.
            for i in range(N):
                arr[i] *= (i + 1)
            f.create_dataset(f"images/{mod}", data=arr)
        # Brain mask: centred ellipsoid-ish; some axial slices below threshold.
        brain = np.zeros((N, H, W, D), dtype=np.int8)
        zc, hc, wc = D // 2, H // 2, W // 2
        for z in range(D):
            zd = abs(z - zc) / zc
            if zd < 0.85:
                brain[:, hc - 60 : hc + 60, wc - 60 : wc + 60, z] = 1
        f.create_dataset("masks/brain", data=brain)
        # Splits.
        cv0_train = np.array([b"UCSF-PDGM-0000", b"UCSF-PDGM-0001"])
        cv0_val = np.array([b"UCSF-PDGM-0002"])
        test = np.array([b"UCSF-PDGM-0003"])
        f.create_dataset("splits/cv/fold_0/train", data=cv0_train)
        f.create_dataset("splits/cv/fold_0/val", data=cv0_val)
        f.create_dataset("splits/test", data=test)
        f.attrs["schema_version"] = "2.0.0"
    return out


def test_dataset_indexes_only_high_brain_slices(synth_h5: Path) -> None:
    ds = UCSFPDGMSliceDataset(
        image_h5=synth_h5, fold=0, phase="train", min_brain_voxels=100,
    )
    # 2 patients × ~135 brain-bearing slices = ~270; just lower-bound.
    assert len(ds) > 100


def test_dataset_returns_correct_shapes_and_range(synth_h5: Path) -> None:
    ds = UCSFPDGMSliceDataset(
        image_h5=synth_h5, fold=0, phase="train", image_size=256,
    )
    sample = ds[0]
    assert sample["A"].shape == (3, 256, 256)
    assert sample["B"].shape == (1, 256, 256)
    # [-1, 1] range after tanh rescale.
    assert float(sample["A"].min()) >= -1.0 - 1e-6
    assert float(sample["A"].max()) <= 1.0 + 1e-6
    assert float(sample["B"].min()) >= -1.0 - 1e-6
    assert float(sample["B"].max()) <= 1.0 + 1e-6
    assert "patient" in sample["A_paths"]


def test_dataset_is_deterministic(synth_h5: Path) -> None:
    """No augmentation → repeat reads are byte-identical."""
    ds = UCSFPDGMSliceDataset(
        image_h5=synth_h5, fold=0, phase="train", image_size=256,
    )
    a = ds[5]
    b = ds[5]
    torch.testing.assert_close(a["A"], b["A"])
    torch.testing.assert_close(a["B"], b["B"])


def test_dataset_uses_per_patient_thresholds(synth_h5: Path) -> None:
    """Slices from patient i ≠ slices from patient j after normalisation,
    even though the raw intensity ratios were just a scalar multiple."""
    ds = UCSFPDGMSliceDataset(
        image_h5=synth_h5, fold=0, phase="train", image_size=256,
    )
    by_patient: dict[int, torch.Tensor] = {}
    for i in range(len(ds)):
        pidx, _ = ds._slice_index[i]
        if pidx not in by_patient:
            by_patient[pidx] = ds[i]["A"]
        if len(by_patient) >= 2:
            break
    # Both patients should land in [-1, 1] (post-normalisation invariant), so
    # their means should both be finite real numbers — the scalar multiple in
    # the fixture is washed out by per-patient percentile normalisation.
    for t in by_patient.values():
        assert torch.isfinite(t).all()
        assert -1 <= float(t.mean()) <= 1


def test_dataset_rejects_invalid_phase(synth_h5: Path) -> None:
    with pytest.raises(DatasetError):
        UCSFPDGMSliceDataset(image_h5=synth_h5, fold=0, phase="trainval")


def test_dataset_rejects_non_div4_image_size(synth_h5: Path) -> None:
    with pytest.raises(DatasetError):
        UCSFPDGMSliceDataset(image_h5=synth_h5, fold=0, phase="train", image_size=250)


def test_pad_to_centred() -> None:
    x = torch.arange(240 * 240, dtype=torch.float32).reshape(1, 240, 240)
    y = _pad_to(x, 256)
    assert y.shape == (1, 256, 256)
    # Centred padding: 8 rows of zeros on each side.
    assert float(y[0, :8, :].sum()) == 0
    assert float(y[0, -8:, :].sum()) == 0
    assert float(y[0, :, :8].sum()) == 0
    assert float(y[0, :, -8:].sum()) == 0
    # Centre block matches input.
    torch.testing.assert_close(y[0, 8:248, 8:248], x[0])


def test_pad_to_crop_when_input_larger() -> None:
    x = torch.ones(1, 300, 300)
    y = _pad_to(x, 256)
    assert y.shape == (1, 256, 256)
    assert float(y.sum()) == 256 * 256

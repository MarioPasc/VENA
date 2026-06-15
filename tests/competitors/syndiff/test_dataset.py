"""Tests for the SynDiff VENA dataset wrapper.

Mirrors the pgan_cgan test set with one structural difference: SynDiff's
dataset returns a 2-tuple ``(x1=target, x2=source)`` (TensorDataset contract
inherited from upstream), not a dict.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vena.competitors.syndiff.dataset import (
    DatasetError,
    SynDiffSliceDataset,
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
            for i in range(N):
                arr[i] *= (i + 1)
            f.create_dataset(f"images/{mod}", data=arr)
        brain = np.zeros((N, H, W, D), dtype=np.int8)
        zc, hc, wc = D // 2, H // 2, W // 2
        for z in range(D):
            zd = abs(z - zc) / zc
            if zd < 0.85:
                brain[:, hc - 60 : hc + 60, wc - 60 : wc + 60, z] = 1
        f.create_dataset("masks/brain", data=brain)
        f.create_dataset("splits/cv/fold_0/train",
                         data=np.array([b"UCSF-PDGM-0000", b"UCSF-PDGM-0001"]))
        f.create_dataset("splits/cv/fold_0/val",
                         data=np.array([b"UCSF-PDGM-0002"]))
        f.create_dataset("splits/test", data=np.array([b"UCSF-PDGM-0003"]))
        f.attrs["schema_version"] = "2.0.0"
    return out


def test_dataset_indexes_only_high_brain_slices(synth_h5: Path) -> None:
    ds = SynDiffSliceDataset(
        image_h5=synth_h5, fold=0, phase="train", min_brain_voxels=100,
    )
    assert len(ds) > 100


def test_dataset_returns_correct_shapes_and_range(synth_h5: Path) -> None:
    ds = SynDiffSliceDataset(
        image_h5=synth_h5, fold=0, phase="train",
        source_modality="t1pre", target_modality="t1c", image_size=256,
    )
    target, source = ds[0]
    # SynDiff is single-source one-to-one — both elements are 1-channel.
    assert target.shape == (1, 256, 256)
    assert source.shape == (1, 256, 256)
    # [-1, 1] range after tanh rescale.
    for x in (target, source):
        assert float(x.min()) >= -1.0 - 1e-6
        assert float(x.max()) <= 1.0 + 1e-6


def test_dataset_is_deterministic(synth_h5: Path) -> None:
    """No augmentation → repeat reads are byte-identical."""
    ds = SynDiffSliceDataset(
        image_h5=synth_h5, fold=0, phase="train", image_size=256,
    )
    a_target, a_source = ds[5]
    b_target, b_source = ds[5]
    torch.testing.assert_close(a_target, b_target)
    torch.testing.assert_close(a_source, b_source)


def test_dataset_rejects_invalid_phase(synth_h5: Path) -> None:
    with pytest.raises(DatasetError):
        SynDiffSliceDataset(image_h5=synth_h5, fold=0, phase="trainval")


def test_dataset_rejects_non_div32_image_size(synth_h5: Path) -> None:
    # SynDiff's 6-level NCSN++ needs image_size % 32 == 0; 256 OK, 250 not.
    with pytest.raises(DatasetError):
        SynDiffSliceDataset(image_h5=synth_h5, fold=0, phase="train", image_size=250)


def test_dataset_rejects_same_source_and_target(synth_h5: Path) -> None:
    with pytest.raises(DatasetError):
        SynDiffSliceDataset(
            image_h5=synth_h5, fold=0, phase="train",
            source_modality="t1c", target_modality="t1c",
        )


def test_pad_to_centred() -> None:
    x = torch.arange(240 * 240, dtype=torch.float32).reshape(1, 240, 240)
    y = _pad_to(x, 256)
    assert y.shape == (1, 256, 256)
    assert float(y[0, :8, :].sum()) == 0
    assert float(y[0, -8:, :].sum()) == 0
    torch.testing.assert_close(y[0, 8:248, 8:248], x[0])


def test_dataset_falls_back_to_flat_splits_schema(tmp_path) -> None:
    """REMBRANDT uses splits/{train,val,test} (no k-fold) — verify fallback."""
    out = tmp_path / "flat.h5"
    H, W, D = 100, 100, 60
    rng = np.random.default_rng(0)
    N = 5
    with h5py.File(out, "w") as f:
        f.create_dataset("ids", data=np.array([f"P-{i:03d}" for i in range(N)], dtype="S10"))
        for mod in ("t1pre", "t1c", "t2", "flair"):
            f.create_dataset(
                f"images/{mod}",
                data=rng.uniform(0.0, 100.0, size=(N, H, W, D)).astype(np.float32),
            )
        f.create_dataset("masks/brain", data=np.ones((N, H, W, D), dtype=np.int8))
        f.create_dataset("splits/train", data=np.array([b"P-000", b"P-001", b"P-002"]))
        f.create_dataset("splits/val", data=np.array([b"P-003"]))
        f.create_dataset("splits/test", data=np.array([b"P-004"]))
    ds = SynDiffSliceDataset(
        image_h5=out, fold=0, phase="train", image_size=128, min_brain_voxels=10,
    )
    assert len(ds.patient_indices) == 3


def test_dataset_resolves_longitudinal_patient_ids(tmp_path) -> None:
    """BraTS-GLI / LUMIERE store scan-level /ids but patient-level splits."""
    out = tmp_path / "longitudinal.h5"
    H, W, D = 100, 100, 60
    rng = np.random.default_rng(0)
    scan_ids = [f"PT-{p:04d}-{s:03d}" for p in range(3) for s in range(2)]
    with h5py.File(out, "w") as f:
        f.create_dataset("ids", data=np.array(scan_ids, dtype="S20"))
        for mod in ("t1pre", "t1c", "t2", "flair"):
            f.create_dataset(
                f"images/{mod}",
                data=rng.uniform(0.0, 100.0, size=(6, H, W, D)).astype(np.float32),
            )
        f.create_dataset("masks/brain", data=np.ones((6, H, W, D), dtype=np.int8))
        f.create_dataset("splits/cv/fold_0/train", data=np.array([b"PT-0000", b"PT-0001"]))
        f.create_dataset("splits/cv/fold_0/val", data=np.array([b"PT-0002"]))
        f.create_dataset("splits/test", data=np.array([b"PT-0000"]))
    ds = SynDiffSliceDataset(
        image_h5=out, fold=0, phase="train", image_size=128, min_brain_voxels=10,
    )
    # 2 requested patients × 2 scans each = 4 resolved entries.
    assert len(ds.patient_indices) == 4
    assert all("-" in pid and len(pid.split("-")[-1]) == 3 for pid in ds.patient_ids)

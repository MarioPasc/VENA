"""Tests for the T1C-RFlow VENA dataset wrapper.

The fixture builds a tiny synthetic latent H5 that matches VENA's latent
schema 2.0.0: ``latents/{t1pre,flair,t1c}`` shaped ``(N, C, h, w, d)``,
``ids`` vlen-str, ``splits/cv/fold_0/{train,val}``, ``splits/test``. No real
data is needed.

Citation: Eidex et al. 2025, arXiv:2509.24194.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vena.competitors.t1c_rflow.dataset import (
    DatasetError,
    T1CRFlowLatentDataset,
)

pytestmark = pytest.mark.unit


# Small enough to keep the test fast; large enough to be representative of
# VENA's actual latent grid (4×60×60×40 on UCSF-PDGM).
_N = 4
_C = 4
_LH, _LW, _LD = 8, 8, 6


def _make_synth_latent_h5(
    out: Path,
    n: int = _N,
    *,
    ids: list[bytes] | None = None,
    longitudinal: bool = False,
    flat_splits: bool = False,
    modalities: tuple[str, ...] = ("t1pre", "flair", "t1c"),
) -> Path:
    """Build a synthetic latent H5 mirroring VENA's schema.

    Parameters
    ----------
    longitudinal : bool
        If True, ``/ids`` holds scan-level ids (``PT-XXXX-NNN``) but splits
        hold patient-level ids (``PT-XXXX``) — the BraTS-GLI / LUMIERE case.
    flat_splits : bool
        If True, write ``splits/{train,val,test}`` instead of the nested
        k-fold layout — the REMBRANDT case.
    """
    rng = np.random.default_rng(0)
    with h5py.File(out, "w") as f:
        if ids is None:
            if longitudinal:
                # 2 scans per patient — PT-0000-000, PT-0000-001, PT-0001-000, ...
                base = [f"PT-{i:04d}" for i in range(n // 2 + n % 2)]
                ids = []
                for b in base:
                    ids.append(f"{b}-000".encode())
                    if len(ids) < n:
                        ids.append(f"{b}-001".encode())
            else:
                ids = [f"COHORT-A-{i:04d}".encode() for i in range(n)]
        f.create_dataset("ids", data=np.asarray(ids))

        for mod in modalities:
            arr = rng.standard_normal(
                size=(n, _C, _LH, _LW, _LD)
            ).astype(np.float32)
            # Force per-patient scale so per-sample distinctness is detectable.
            for i in range(n):
                arr[i] *= (i + 1) * 0.1
            f.create_dataset(f"latents/{mod}", data=arr)

        if flat_splits:
            train_ids = [ids[0], ids[1]]
            val_ids = [ids[2]] if len(ids) > 2 else []
            test_ids = [ids[3]] if len(ids) > 3 else []
            f.create_dataset("splits/train", data=np.asarray(train_ids))
            if val_ids:
                f.create_dataset("splits/val", data=np.asarray(val_ids))
            if test_ids:
                f.create_dataset("splits/test", data=np.asarray(test_ids))
        else:
            if longitudinal:
                # Patient-level — half as many as scan-level ids.
                pats = sorted(
                    {s.decode().rsplit("-", 1)[0] for s in ids}
                )
                # First 1 patient → train, last → val/test if any.
                train = [pats[0].encode()]
                val = [pats[1].encode()] if len(pats) > 1 else []
                test = [pats[2].encode()] if len(pats) > 2 else []
            else:
                train = [ids[0], ids[1]] if len(ids) > 1 else [ids[0]]
                val = [ids[2]] if len(ids) > 2 else []
                test = [ids[3]] if len(ids) > 3 else []
            f.create_dataset(
                "splits/cv/fold_0/train", data=np.asarray(train)
            )
            if val:
                f.create_dataset(
                    "splits/cv/fold_0/val", data=np.asarray(val)
                )
            if test:
                f.create_dataset("splits/test", data=np.asarray(test))

        f.attrs["schema_version"] = "2.0.0"
    return out


@pytest.fixture
def synth_h5(tmp_path: Path) -> Path:
    return _make_synth_latent_h5(tmp_path / "synth_latent.h5")


def test_dataset_returns_expected_keys_and_shapes(synth_h5: Path) -> None:
    ds = T1CRFlowLatentDataset(latent_h5=synth_h5, fold=0, phase="train")
    sample = ds[0]
    assert isinstance(sample["patient_id"], str)
    for k in ("z_t1pre", "z_flair", "z_t1c"):
        assert k in sample, f"missing {k!r}"
        assert isinstance(sample[k], torch.Tensor)
        assert sample[k].shape == (_C, _LH, _LW, _LD)
        assert sample[k].dtype == torch.float32


def test_dataset_length_equals_train_split(synth_h5: Path) -> None:
    ds = T1CRFlowLatentDataset(latent_h5=synth_h5, fold=0, phase="train")
    assert len(ds) == 2  # 2 train ids in the synth fixture


def test_dataset_is_deterministic(synth_h5: Path) -> None:
    """Repeat reads of the same index return byte-identical tensors.

    Pins the no-augmentation contract — if anyone introduces a transform in
    ``__getitem__``, this test fails.
    """
    ds = T1CRFlowLatentDataset(latent_h5=synth_h5, fold=0, phase="train")
    a = ds[0]
    b = ds[0]
    for k in ("z_t1pre", "z_flair", "z_t1c"):
        torch.testing.assert_close(a[k], b[k], rtol=0.0, atol=0.0)


def test_dataset_per_patient_distinctness(synth_h5: Path) -> None:
    """Each patient yields a distinct sample (no accidental aliasing)."""
    ds = T1CRFlowLatentDataset(latent_h5=synth_h5, fold=0, phase="train")
    a = ds[0]["z_t1pre"]
    b = ds[1]["z_t1pre"]
    assert not torch.allclose(a, b)


def test_dataset_rejects_invalid_phase(synth_h5: Path) -> None:
    with pytest.raises(DatasetError):
        T1CRFlowLatentDataset(latent_h5=synth_h5, fold=0, phase="trainval")


def test_dataset_rejects_missing_latent_dataset(tmp_path: Path) -> None:
    out = _make_synth_latent_h5(
        tmp_path / "no_swan.h5", modalities=("t1pre", "t1c")  # no flair
    )
    with pytest.raises(DatasetError, match="flair"):
        T1CRFlowLatentDataset(latent_h5=out, fold=0, phase="train")


def test_dataset_falls_back_to_flat_splits_schema(tmp_path: Path) -> None:
    """REMBRANDT-style: ``splits/{train,val,test}`` without k-fold."""
    out = _make_synth_latent_h5(tmp_path / "flat.h5", flat_splits=True)
    ds = T1CRFlowLatentDataset(latent_h5=out, fold=0, phase="train")
    assert len(ds) == 2
    ds_val = T1CRFlowLatentDataset(latent_h5=out, fold=0, phase="val")
    assert len(ds_val) == 1


def test_dataset_resolves_longitudinal_patient_ids(tmp_path: Path) -> None:
    """BraTS-GLI/LUMIERE-style: scan ids in /ids, patient ids in splits."""
    out = _make_synth_latent_h5(
        tmp_path / "long.h5",
        n=6,
        longitudinal=True,
    )
    ds = T1CRFlowLatentDataset(latent_h5=out, fold=0, phase="train")
    # Patient PT-0000 has 2 scans → 2 dataset entries from one split id.
    assert len(ds) == 2


def test_dataset_max_patients_caps_split(synth_h5: Path) -> None:
    ds = T1CRFlowLatentDataset(
        latent_h5=synth_h5, fold=0, phase="train", max_patients=1
    )
    assert len(ds) == 1


def test_dataset_pickle_drops_h5_handle(synth_h5: Path) -> None:
    """h5py handles are not picklable — ``__getstate__`` must drop them."""
    import pickle

    ds = T1CRFlowLatentDataset(latent_h5=synth_h5, fold=0, phase="train")
    _ = ds[0]
    assert ds._h5 is not None  # opened by __getitem__

    state = ds.__getstate__()
    assert state["_h5"] is None  # handle dropped at pickle time

    # Round-trip via pickle and verify the dataset still works.
    blob = pickle.dumps(ds)
    ds2 = pickle.loads(blob)
    s = ds2[0]
    assert s["z_t1pre"].shape == (_C, _LH, _LW, _LD)

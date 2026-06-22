"""S1 v3 batch-layout regression tests.

Pins the contract that ``LatentH5Dataset._read_one`` (and the aug parallel
path) emit ``m_tumor`` (3-channel raw soft mask) plus per-class slices
``m_netc``, ``m_ed``, ``m_et`` alongside the existing single-channel ``m_wt``.

Back-compat invariant: ``m_wt`` keeps its sum-then-threshold derivation, so
contrastive v0.4 numerics are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

from vena.model.fm.lightning.data import LatentH5Dataset

_LATENT_SHAPE = (4, 4, 4)  # (h, w, d) — tiny for fast tests
_N_LATENT_CHANNELS = 4
_N_MASK_CHANNELS = 3
_WT_THRESHOLD = 0.5


def _build_minimal_latent_h5(path: Path, patient_ids: list[str]) -> None:
    """Write a minimal schema-2.0.0 latent H5 to *path*.

    One scan per patient. Random soft per-class tumour mask, random latents.
    No brain mask (mirrors legacy schema-2.0.0 H5s).
    """
    n_scans = len(patient_ids)
    rng = np.random.default_rng(42)
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0.0"
        f.attrs["cohort"] = "TestCohort"
        f.attrs["created_at"] = "2026-06-22T00:00:00Z"
        f.attrs["producer"] = "test_v3_batch_layout"
        f.attrs["config_json"] = "{}"
        f.attrs["git_sha"] = "deadbeef"
        f.create_dataset("ids", data=np.array(patient_ids, dtype=object), dtype=dt)
        for mod in ("t1pre", "t1c", "t2", "flair"):
            data = rng.random((n_scans, _N_LATENT_CHANNELS, *_LATENT_SHAPE), dtype=np.float32)
            f.create_dataset(f"latents/{mod}", data=data, compression="gzip")
        masks = rng.random((n_scans, _N_MASK_CHANNELS, *_LATENT_SHAPE), dtype=np.float32)
        f.create_dataset("masks/tumor_latent", data=masks, compression="gzip")


def test_v3_batch_carries_m_tumor_and_per_class_slices(tmp_path: Path) -> None:
    """`m_tumor`, `m_netc`, `m_ed`, `m_et` must all be present in every item."""
    h5 = tmp_path / "tiny_latents.h5"
    pids = ["P0", "P1"]
    _build_minimal_latent_h5(h5, pids)

    ds = LatentH5Dataset(h5, pids, wt_threshold=_WT_THRESHOLD)
    item = ds[0]
    for key in ("m_wt", "m_tumor", "m_netc", "m_ed", "m_et"):
        assert key in item, f"missing batch key {key!r}"

    assert item["m_tumor"].shape == torch.Size([_N_MASK_CHANNELS, *_LATENT_SHAPE])
    assert item["m_wt"].shape == torch.Size([1, *_LATENT_SHAPE])
    for key in ("m_netc", "m_ed", "m_et"):
        assert item[key].shape == torch.Size([1, *_LATENT_SHAPE]), (
            f"{key} shape mismatch: {item[key].shape}"
        )


def test_v3_per_class_slices_match_m_tumor_channels(tmp_path: Path) -> None:
    """The single-channel slices must equal the corresponding `m_tumor` channels."""
    h5 = tmp_path / "tiny_latents.h5"
    pids = ["P0"]
    _build_minimal_latent_h5(h5, pids)

    ds = LatentH5Dataset(h5, pids, wt_threshold=_WT_THRESHOLD)
    item = ds[0]
    m_tumor = item["m_tumor"]
    torch.testing.assert_close(item["m_netc"], m_tumor[0:1])
    torch.testing.assert_close(item["m_ed"], m_tumor[1:2])
    torch.testing.assert_close(item["m_et"], m_tumor[2:3])


def test_v3_m_wt_preserves_sum_then_threshold_semantics(tmp_path: Path) -> None:
    """Adding m_tumor must not change the m_wt derivation.

    The contrastive v0.4 loss path consumes m_wt; any change in semantics
    would shift its numerics. Spec doc §2.4 proposes `max`-then-threshold,
    but the v3 scope decision is to keep the current sum-then-threshold to
    preserve back-compat.
    """
    h5 = tmp_path / "tiny_latents.h5"
    pids = ["P0"]
    _build_minimal_latent_h5(h5, pids)

    ds = LatentH5Dataset(h5, pids, wt_threshold=_WT_THRESHOLD)
    item = ds[0]
    # Reproduce the expected m_wt from m_tumor (same formula as _read_one).
    soft_union = item["m_tumor"].sum(dim=0, keepdim=True).clamp(0.0, 1.0)
    expected_m_wt = (soft_union >= _WT_THRESHOLD).float()
    torch.testing.assert_close(item["m_wt"], expected_m_wt)

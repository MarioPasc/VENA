"""Tests for ``recompute_union_of_four`` — Phase 1.1 of 2026-06-19 fix-up.

Covers the helper used by ``scripts/harmonize_brain_source_inplace.py`` to
unify BraTS-GLI + IvyGAP brain masks (currently ``t1pre > 0``) onto the
union-of-4-modalities policy used by every other VENA-computed cohort.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.h5.shared.brain_mask import recompute_union_of_four

pytestmark = pytest.mark.unit


def _make_image_h5(path: Path) -> Path:
    """Plant two patients; one with non-overlapping nonzero supports per modality."""
    shape = (20, 20, 20)
    # Patient 0: t1pre carries voxels in the left half only; t2 / flair / t1c
    # cover the right half. Old brain mask = t1pre > 0 = left half. Union of
    # four would cover the whole volume.
    images = {
        "t1pre": np.zeros((2, *shape), dtype=np.float32),
        "t1c": np.zeros((2, *shape), dtype=np.float32),
        "t2": np.zeros((2, *shape), dtype=np.float32),
        "flair": np.zeros((2, *shape), dtype=np.float32),
    }
    images["t1pre"][0, :, :10, :] = 1.0
    images["t1c"][0, :, 10:, :] = 1.0
    images["t2"][0, :, 10:, :] = 1.0
    images["flair"][0, :, 10:, :] = 1.0
    # Patient 1: every modality covers the whole interior (no change expected).
    for slug in images:
        images[slug][1, 2:18, 2:18, 2:18] = 1.0
    old_brain = np.zeros((2, *shape), dtype=np.int8)
    old_brain[0, :, :10, :] = 1  # legacy t1pre-only
    old_brain[1, 2:18, 2:18, 2:18] = 1  # already correct
    str_dt = h5py.string_dtype()
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "ids",
            data=np.array(["P-A", "P-B"], dtype=object),
            dtype=str_dt,
        )
        for slug, arr in images.items():
            f.create_dataset(f"images/{slug}", data=arr, dtype=np.float32)
        f.create_dataset("masks/brain", data=old_brain, dtype=np.int8)
    return path


def test_yields_old_and_new_per_row(tmp_path: Path) -> None:
    p = _make_image_h5(tmp_path / "x.h5")
    rows = list(recompute_union_of_four(p))
    assert len(rows) == 2
    row0_idx, pid0, old0, new0 = rows[0]
    assert row0_idx == 0
    assert pid0 == "P-A"
    assert old0.sum() < new0.sum()  # union grew the mask
    assert new0.sum() > 0
    row1_idx, pid1, old1, new1 = rows[1]
    assert row1_idx == 1
    assert pid1 == "P-B"
    # Patient B was already correct; union should match (modulo CC clean).
    assert new1.sum() == old1.sum()


def test_keeps_dtype(tmp_path: Path) -> None:
    p = _make_image_h5(tmp_path / "x.h5")
    for _row, _pid, _old, new in recompute_union_of_four(p):
        assert new.dtype == np.int8


def test_raises_when_modality_missing(tmp_path: Path) -> None:
    p = _make_image_h5(tmp_path / "x.h5")
    # Drop one modality.
    with h5py.File(p, "r+") as f:
        del f["images/flair"]
    with pytest.raises(KeyError):
        list(recompute_union_of_four(p))


def test_raises_when_brain_missing(tmp_path: Path) -> None:
    p = _make_image_h5(tmp_path / "x.h5")
    with h5py.File(p, "r+") as f:
        del f["masks/brain"]
    with pytest.raises(KeyError):
        list(recompute_union_of_four(p))

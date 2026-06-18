"""Tests for ``vena.data.h5.shared.splits.normalize_splits``.

Covers the 2026-06-19 audit §6.2 schema-unification helper that drops
legacy ``splits/{train,val,test}`` flat aliases on cv cohorts and
``splits/cv/fold_0/{train,val}`` aliases on test-only cohorts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.h5.shared.splits import normalize_splits

pytestmark = pytest.mark.unit


def _make_h5(path: Path, *, role: str, with_flat: bool, with_cv_alias: bool) -> Path:
    str_dt = h5py.string_dtype()
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "ids", data=np.array(["P-001", "P-002", "P-003"], dtype=object), dtype=str_dt
        )
        f.create_dataset("splits/test", data=np.array(["P-003"], dtype=object), dtype=str_dt)
        if role == "cv":
            f.create_dataset(
                "splits/cv/fold_0/train", data=np.array(["P-001"], dtype=object), dtype=str_dt
            )
            f.create_dataset(
                "splits/cv/fold_0/val", data=np.array(["P-002"], dtype=object), dtype=str_dt
            )
            if with_flat:
                f.create_dataset(
                    "splits/train", data=np.array(["P-001"], dtype=object), dtype=str_dt
                )
                f.create_dataset("splits/val", data=np.array(["P-002"], dtype=object), dtype=str_dt)
        elif role == "test_only" and with_cv_alias:
            f.create_dataset(
                "splits/cv/fold_0/val", data=np.array(["P-003"], dtype=object), dtype=str_dt
            )
            f.create_dataset(
                "splits/cv/fold_0/train", data=np.array([], dtype=object), dtype=str_dt
            )
    return path


def test_cv_role_drops_flat_keeps_test_and_cv(tmp_path: Path) -> None:
    p = _make_h5(tmp_path / "x.h5", role="cv", with_flat=True, with_cv_alias=False)
    out = normalize_splits(p, role="cv")
    assert sorted(out["removed"]) == ["splits/train", "splits/val"]
    with h5py.File(p, "r") as f:
        assert "splits/train" not in f
        assert "splits/val" not in f
        assert "splits/test" in f
        assert "splits/cv/fold_0/train" in f
        assert "splits/cv/fold_0/val" in f


def test_test_only_role_drops_cv_subtree(tmp_path: Path) -> None:
    p = _make_h5(tmp_path / "x.h5", role="test_only", with_flat=False, with_cv_alias=True)
    out = normalize_splits(p, role="test_only")
    assert out["removed"] == ["splits/cv"]
    with h5py.File(p, "r") as f:
        assert "splits/cv" not in f
        assert "splits/test" in f


def test_dry_run_does_not_mutate(tmp_path: Path) -> None:
    p = _make_h5(tmp_path / "x.h5", role="cv", with_flat=True, with_cv_alias=False)
    out = normalize_splits(p, role="cv", dry_run=True)
    assert sorted(out["removed"]) == ["splits/train", "splits/val"]
    with h5py.File(p, "r") as f:
        assert "splits/train" in f
        assert "splits/val" in f


def test_idempotent_when_already_canonical(tmp_path: Path) -> None:
    p = _make_h5(tmp_path / "x.h5", role="cv", with_flat=False, with_cv_alias=False)
    out_a = normalize_splits(p, role="cv")
    out_b = normalize_splits(p, role="cv")
    assert out_a["removed"] == []
    assert out_b["removed"] == []


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        normalize_splits(tmp_path / "nope.h5", role="cv")

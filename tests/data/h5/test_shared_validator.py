"""Validator pair against a hand-built mini H5."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.h5.shared import (
    DatasetSpec,
    H5Manifest,
    H5Writer,
    assert_h5_valid,
    validate_h5,
)
from vena.data.h5.shared.exceptions import H5ValidationError


def _mini_manifest() -> H5Manifest:
    return H5Manifest(
        schema_version="0.1.0",
        cohort="MINI",
        domain="image",
        expected_shape=(4, 4, 4),
        datasets=[
            DatasetSpec(
                path="ids",
                dtype="vlen-str",
                kind="id",
                units="dimensionless",
                description="id",
                leading_dim="n_scans",
            ),
            DatasetSpec(
                path="images/x",
                dtype="float32",
                kind="image",
                units="au",
                description="x",
                leading_dim="n_scans",
            ),
            DatasetSpec(
                path="masks/m",
                dtype="int8",
                kind="mask",
                units="label",
                description="m",
                leading_dim="n_scans",
            ),
        ],
    )


def _build_mini_h5(path: Path, n: int = 3) -> None:
    manifest = _mini_manifest()
    with H5Writer(
        path,
        manifest=manifest,
        config_json="{}",
        producer="test:0",
        created_at="2026-01-01T00:00:00Z",
        git_sha=None,
        overwrite=True,
    ) as w:
        ids = w.create_1d(manifest.get("ids"), n=n)
        ids[:] = np.asarray([f"P{i}" for i in range(n)], dtype=object)
        img = w.create_stacked(manifest.get("images/x"), n=n, spatial_shape=(4, 4, 4))
        img[...] = np.zeros((n, 4, 4, 4), dtype=np.float32)
        msk = w.create_stacked(manifest.get("masks/m"), n=n, spatial_shape=(4, 4, 4))
        msk[...] = np.zeros((n, 4, 4, 4), dtype=np.int8)


@pytest.mark.unit
def test_valid_file_passes(tmp_path: Path) -> None:
    p = tmp_path / "mini.h5"
    _build_mini_h5(p)
    assert validate_h5(p, _mini_manifest()) == []
    assert_h5_valid(p, _mini_manifest())


@pytest.mark.unit
def test_missing_dataset_is_reported(tmp_path: Path) -> None:
    p = tmp_path / "mini.h5"
    _build_mini_h5(p)
    with h5py.File(p, "a") as f:
        del f["masks/m"]
    violations = validate_h5(p, _mini_manifest())
    assert any("masks/m" in v for v in violations)
    with pytest.raises(H5ValidationError):
        assert_h5_valid(p, _mini_manifest())


@pytest.mark.unit
def test_dtype_mismatch_is_reported(tmp_path: Path) -> None:
    p = tmp_path / "mini.h5"
    _build_mini_h5(p)
    # Replace the image dataset with a float64 copy of the same shape.
    with h5py.File(p, "a") as f:
        data = f["images/x"][...].astype(np.float64)
        attrs = dict(f["images/x"].attrs)
        del f["images/x"]
        d = f.create_dataset("images/x", data=data, dtype=np.float64)
        for k, v in attrs.items():
            d.attrs[k] = v
    violations = validate_h5(p, _mini_manifest())
    assert any("dtype" in v for v in violations)


@pytest.mark.unit
def test_schema_version_mismatch_is_reported(tmp_path: Path) -> None:
    p = tmp_path / "mini.h5"
    _build_mini_h5(p)
    bumped = _mini_manifest().model_copy(update={"schema_version": "0.2.0"})
    violations = validate_h5(p, bumped)
    assert any("schema_version" in v for v in violations)

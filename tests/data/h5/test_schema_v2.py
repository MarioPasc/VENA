"""Schema-2.0.0 contract: v2 root attrs are required, CSR datasets opt out of n_scans."""

from __future__ import annotations

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
from vena.data.h5.shared.validator import _REQUIRED_ROOT_ATTRS

_V2_ATTRS = ("split_role", "longitudinal", "label_system", "crop_box", "orientation")


def _manifest() -> H5Manifest:
    return H5Manifest(
        schema_version="2.0.0",
        cohort="TEST",
        domain="image",
        expected_shape=(4, 4, 4),
        datasets=[
            DatasetSpec(path="ids", dtype="vlen-str", kind="id", units="dimensionless",
                        description="ids", leading_dim="n_scans"),
            DatasetSpec(path="images/x", dtype="float32", kind="image", units="intensity_au",
                        description="x", leading_dim="n_scans"),
            # CSR datasets: lengths are n_patients(+1), NOT n_scans → leading_dim=None.
            DatasetSpec(path="patients/offsets", dtype="int32", kind="metadata",
                        units="dimensionless", description="csr offsets", leading_dim=None),
            DatasetSpec(path="patients/keys", dtype="vlen-str", kind="id",
                        units="dimensionless", description="csr keys", leading_dim=None),
        ],
    )


def _write(path, manifest, *, extra_root_attrs=None) -> None:
    with H5Writer(path, manifest=manifest, config_json="{}", producer="test:0",
                  created_at="2026-01-01T00:00:00Z", git_sha="deadbeef",
                  extra_root_attrs=extra_root_attrs, overwrite=True) as w:
        ids = w.create_1d(manifest.get("ids"), n=3)
        ids[:] = np.asarray(["a", "b", "c"], dtype=object)
        img = w.create_stacked(manifest.get("images/x"), n=3, spatial_shape=(4, 4, 4))
        img[:] = np.zeros((3, 4, 4, 4), dtype=np.float32)
        # CSR with a length (n_patients=2 → offsets len 3) that is NOT n_scans=3.
        w.write_int_1d("patients/offsets", np.array([0, 2, 3], dtype=np.int32), dtype="int32")
        w.write_vlen_str_1d("patients/keys", ["pa", "pb"], description="keys")


@pytest.mark.unit
def test_v2_attrs_are_required() -> None:
    for a in _V2_ATTRS:
        assert a in _REQUIRED_ROOT_ATTRS


@pytest.mark.unit
def test_writer_stamps_v2_defaults_and_validates(tmp_path) -> None:
    p = tmp_path / "v2.h5"
    m = _manifest()
    _write(p, m)
    assert validate_h5(p, m) == []  # passes with stamped defaults
    with h5py.File(p, "r") as f:
        assert f.attrs["orientation"] == "unknown"  # default
        assert bool(f.attrs["longitudinal"]) is False
        # CSR offsets length (3) != n_scans (3 here by coincidence) — use keys (2).
        assert f["patients/keys"].shape[0] == 2  # not n_scans → must still validate


@pytest.mark.unit
def test_extra_root_attrs_override(tmp_path) -> None:
    p = tmp_path / "v2b.h5"
    m = _manifest()
    _write(p, m, extra_root_attrs={"orientation": "LPS", "longitudinal": True,
                                   "label_system": "BraTS2023", "crop_box": "[192, 224, 192]",
                                   "split_role": "test_only"})
    with h5py.File(p, "r") as f:
        assert f.attrs["orientation"] == "LPS"
        assert bool(f.attrs["longitudinal"]) is True
        assert f.attrs["label_system"] == "BraTS2023"


@pytest.mark.unit
def test_missing_v2_attr_fails_validation(tmp_path) -> None:
    p = tmp_path / "v2c.h5"
    m = _manifest()
    _write(p, m)
    with h5py.File(p, "r+") as f:
        del f.attrs["orientation"]
    violations = validate_h5(p, m)
    assert any("orientation" in v for v in violations)
    with pytest.raises(Exception):
        assert_h5_valid(p, m)


@pytest.mark.unit
def test_csr_leading_dim_none_skips_n_scans_check(tmp_path) -> None:
    """A CSR dataset whose length != n_scans must NOT raise a leading-dim violation."""
    p = tmp_path / "v2d.h5"
    m = _manifest()
    _write(p, m)
    # patients/offsets has length 3 here; make n_scans unambiguous by checking
    # the validator does not complain about patients/* despite leading_dim=None.
    violations = validate_h5(p, m)
    assert not any(v.startswith("patients/") for v in violations), violations

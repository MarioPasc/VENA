"""Tests for the conditional ``masks/brain_latent`` check in ``validate_aug_latent_h5``.

When the producer records the brain-to-latent post-pass with the root attr
``produced_by_brain_to_latent = True``, the validator must fail-fast on a
missing or mis-shaped ``masks/brain_latent``. When the attr is absent or
False (legacy pre-2026-06-19 files), the dataset is optional.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.h5.augmented.latent_domain import (
    AUG_LATENT_SCHEMA_VERSION,
    build_aug_latent_manifest,
    validate_aug_latent_h5,
)
from vena.data.h5.latent_domain.manifest import LATENT_CHANNELS, LATENT_SPATIAL

pytestmark = pytest.mark.unit


def _make_valid_aug_latent(path: Path, *, with_brain_latent: bool, brain_shape=None) -> Path:
    """Minimal aug-latent H5 conforming to the manifest sans the brain_latent."""
    str_dt = h5py.string_dtype()
    n = 2
    manifest = build_aug_latent_manifest(
        cohort="Test", modalities=["t1pre", "t1c"], mask_output_channels=3
    )
    with h5py.File(path, "w") as f:
        # Root attrs required by validate_h5 + aug validator.
        f.attrs["schema_version"] = AUG_LATENT_SCHEMA_VERSION
        f.attrs["cohort"] = "Test"
        f.attrs["domain"] = "latent"
        f.attrs["created_at"] = "1970-01-01T00:00:00Z"
        f.attrs["producer"] = "test"
        f.attrs["config_json"] = "{}"
        f.attrs["manifest_json"] = manifest.to_json()
        f.attrs["git_sha"] = "deadbeef"
        f.attrs["split_role"] = "cv"
        f.attrs["longitudinal"] = False
        f.attrs["label_system"] = "BraTS2021"
        f.attrs["crop_box"] = json.dumps([192, 224, 192])
        f.attrs["orientation"] = "LPS"
        f.attrs["source_aug_image_h5_path"] = "/dev/null"
        f.attrs["source_aug_image_h5_sha256"] = "0" * 64
        f.attrs["aug_config_sha256"] = "0" * 64
        f.attrs["variants_json"] = json.dumps(["v1"])
        # Datasets.
        for path_, spec in [
            ("ids", "vlen-str"),
            ("source_row_index", "int32"),
            ("variants", "vlen-str"),
            ("aug_params_json", "vlen-str"),
        ]:
            if spec == "vlen-str":
                dset = f.create_dataset(
                    path_,
                    data=np.array(["a", "b"], dtype=object)
                    if path_ != "aug_params_json"
                    else np.array(["{}", "{}"], dtype=object),
                    dtype=str_dt,
                )
            else:
                dset = f.create_dataset(path_, data=np.array([0, 1], dtype=np.int32), dtype="int32")
            dset.attrs["units"] = "dimensionless"
            dset.attrs["description"] = "test"
            dset.attrs["dtype"] = spec
        for slug in ("t1pre", "t1c"):
            dset = f.create_dataset(
                f"latents/{slug}",
                data=np.zeros((n, LATENT_CHANNELS, *LATENT_SPATIAL), dtype=np.float32),
                dtype="float32",
            )
            dset.attrs["units"] = "latent_au"
            dset.attrs["description"] = "test"
            dset.attrs["dtype"] = "float32"
        tumor = f.create_dataset(
            "masks/tumor_latent",
            data=np.zeros((n, 3, *LATENT_SPATIAL), dtype=np.float32),
            dtype="float32",
        )
        tumor.attrs["units"] = "dimensionless"
        tumor.attrs["description"] = "test"
        tumor.attrs["dtype"] = "float32"

        if with_brain_latent:
            shape = brain_shape if brain_shape is not None else (n, 1, *LATENT_SPATIAL)
            f.attrs["produced_by_brain_to_latent"] = True
            brain = f.create_dataset(
                "masks/brain_latent",
                data=np.zeros(shape, dtype=np.int8),
                dtype="int8",
            )
            brain.attrs["units"] = "binary"
            brain.attrs["description"] = "test"
            brain.attrs["dtype"] = "int8"
    return path


def test_no_brain_latent_no_flag_passes(tmp_path: Path) -> None:
    p = _make_valid_aug_latent(tmp_path / "x.h5", with_brain_latent=False)
    violations = validate_aug_latent_h5(
        p, cohort="Test", modalities=["t1pre", "t1c"], mask_output_channels=3
    )
    assert violations == []


def test_flag_set_brain_latent_present_passes(tmp_path: Path) -> None:
    p = _make_valid_aug_latent(tmp_path / "x.h5", with_brain_latent=True)
    violations = validate_aug_latent_h5(
        p, cohort="Test", modalities=["t1pre", "t1c"], mask_output_channels=3
    )
    assert violations == []


def test_flag_set_brain_latent_missing_fails(tmp_path: Path) -> None:
    p = _make_valid_aug_latent(tmp_path / "x.h5", with_brain_latent=False)
    with h5py.File(p, "r+") as f:
        f.attrs["produced_by_brain_to_latent"] = True
    violations = validate_aug_latent_h5(
        p, cohort="Test", modalities=["t1pre", "t1c"], mask_output_channels=3
    )
    assert any("masks/brain_latent" in v and "missing" in v for v in violations)


def test_flag_set_brain_latent_bad_shape_fails(tmp_path: Path) -> None:
    p = _make_valid_aug_latent(
        tmp_path / "x.h5",
        with_brain_latent=True,
        brain_shape=(2, 1, 16, 16, 16),  # wrong spatial shape
    )
    violations = validate_aug_latent_h5(
        p, cohort="Test", modalities=["t1pre", "t1c"], mask_output_channels=3
    )
    assert any("masks/brain_latent" in v and "shape" in v for v in violations)

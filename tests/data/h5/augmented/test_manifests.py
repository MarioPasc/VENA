"""Unit tests for the offline-aug H5 manifest validators."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vena.data.h5.augmented import (
    AUG_IMAGE_CROP_BOX,
    AUG_IMAGE_SCHEMA_VERSION,
    AUG_LATENT_SCHEMA_VERSION,
    assert_aug_image_h5_valid,
    assert_aug_latent_h5_valid,
    build_aug_image_manifest,
    build_aug_latent_manifest,
    validate_aug_image_h5,
)
from vena.data.h5.shared import H5Writer
from vena.data.h5.shared.exceptions import H5ValidationError

pytestmark = pytest.mark.unit

_COHORT = "UCSF-PDGM"
_MODS = ["t1pre", "t1c", "t2", "flair"]


def _write_minimal_aug_image_h5(
    path: Path, *, n_rows: int = 4, crop_origin_zero: bool = True
) -> None:
    manifest = build_aug_image_manifest(_COHORT, _MODS)
    extra = {
        "split_role": "cv",
        "longitudinal": False,
        "label_system": "BraTS2021",
        "crop_box": json.dumps(list(AUG_IMAGE_CROP_BOX)),
        "orientation": "LPS",
    }
    with H5Writer(
        path=path,
        manifest=manifest,
        config_json="{}",
        producer="test",
        created_at="2026-06-03T00:00:00Z",
        git_sha="testsha",
        extra_root_attrs=extra,
        overwrite=True,
    ) as w:
        f = w.file
        f.attrs["source_image_h5_path"] = "/tmp/source.h5"
        f.attrs["source_image_h5_sha256"] = "abc"
        f.attrs["aug_config_json"] = "{}"
        f.attrs["aug_config_sha256"] = "deadbeef"
        f.attrs["variants_json"] = json.dumps(["v1", "v2", "v3", "v4"])
        f.attrs["seed"] = 42
        f.attrs["world_size"] = 1
        f.attrs["rank"] = 0

        ids_dset = w.create_1d(manifest.get("ids"), n=n_rows)
        ids_dset[:] = np.asarray([f"PAT{i:03d}" for i in range(n_rows)], dtype=object)
        srci = w.create_1d(manifest.get("source_row_index"), n=n_rows)
        srci[:] = np.arange(n_rows, dtype=np.int32)
        var = w.create_1d(manifest.get("variants"), n=n_rows)
        var[:] = np.asarray(["v1"] * n_rows, dtype=object)
        ap = w.create_1d(manifest.get("aug_params_json"), n=n_rows)
        ap[:] = np.asarray(["{}"] * n_rows, dtype=object)
        for slug in _MODS:
            d = w.create_stacked(
                manifest.get(f"images/{slug}"), n=n_rows, spatial_shape=AUG_IMAGE_CROP_BOX
            )
            d[:] = np.zeros((n_rows, *AUG_IMAGE_CROP_BOX), dtype=np.float32)
        mdset = w.create_stacked(
            manifest.get("masks/tumor"), n=n_rows, spatial_shape=AUG_IMAGE_CROP_BOX
        )
        mdset[:] = np.zeros((n_rows, *AUG_IMAGE_CROP_BOX), dtype=np.int8)

        crop_origin = f.create_dataset("crop/origin", shape=(n_rows, 3), dtype=np.int32)
        crop_origin[:] = 0 if crop_origin_zero else 7
        crop_origin.attrs["units"] = "voxels"
        crop_origin.attrs["description"] = "Per-row crop origin."
        crop_origin.attrs["dtype"] = "int32"


def test_build_aug_image_manifest_round_trip() -> None:
    m = build_aug_image_manifest(_COHORT, _MODS)
    assert m.schema_version == AUG_IMAGE_SCHEMA_VERSION
    assert m.cohort == _COHORT
    assert m.domain == "image"
    assert m.expected_shape == AUG_IMAGE_CROP_BOX
    paths = {d.path for d in m.datasets}
    assert {"ids", "source_row_index", "variants", "aug_params_json", "masks/tumor"} <= paths
    for slug in _MODS:
        assert f"images/{slug}" in paths
    assert m == type(m).from_json(m.to_json())


def test_build_aug_latent_manifest_round_trip() -> None:
    m = build_aug_latent_manifest(_COHORT, _MODS, mask_output_channels=3)
    assert m.schema_version == AUG_LATENT_SCHEMA_VERSION
    assert m.domain == "latent"
    paths = {d.path for d in m.datasets}
    assert {"ids", "source_row_index", "variants", "aug_params_json", "masks/tumor_latent"} <= paths
    for slug in _MODS:
        assert f"latents/{slug}" in paths
    # patients/* and splits/* must NOT appear in the aug-latent schema.
    assert not any(p.startswith("patients/") or p.startswith("splits/") for p in paths)


def test_validate_aug_image_h5_passes(tmp_path: Path) -> None:
    p = tmp_path / "x_image_aug.h5"
    _write_minimal_aug_image_h5(p, n_rows=4, crop_origin_zero=True)
    assert validate_aug_image_h5(p, _COHORT, _MODS) == []
    assert_aug_image_h5_valid(p, _COHORT, _MODS)


def test_validate_aug_image_h5_rejects_nonzero_crop_origin(tmp_path: Path) -> None:
    p = tmp_path / "x_image_aug.h5"
    _write_minimal_aug_image_h5(p, n_rows=4, crop_origin_zero=False)
    with pytest.raises(H5ValidationError, match="crop/origin"):
        assert_aug_image_h5_valid(p, _COHORT, _MODS)


def test_validate_aug_image_h5_rejects_missing_provenance(tmp_path: Path) -> None:
    p = tmp_path / "x_image_aug.h5"
    _write_minimal_aug_image_h5(p, n_rows=4)
    # Now strip an aug-specific root attr after the fact and expect a fail.
    with h5py.File(p, "r+") as f:
        del f.attrs["aug_config_sha256"]
    with pytest.raises(H5ValidationError, match="aug_config_sha256"):
        assert_aug_image_h5_valid(p, _COHORT, _MODS)


def test_validate_aug_latent_h5_rejects_splits_group(tmp_path: Path) -> None:
    """Aug-latent H5 must not carry splits — that lives on the clean H5."""
    p = tmp_path / "x_latents_aug.h5"
    n_rows = 4
    manifest = build_aug_latent_manifest(_COHORT, _MODS, mask_output_channels=3)
    extra = {
        "split_role": "cv",
        "longitudinal": False,
        "label_system": "BraTS2021",
        "crop_box": json.dumps(list(AUG_IMAGE_CROP_BOX)),
        "orientation": "LPS",
    }
    with H5Writer(
        path=p,
        manifest=manifest,
        config_json="{}",
        producer="test",
        created_at="2026-06-03T00:00:00Z",
        git_sha="testsha",
        extra_root_attrs=extra,
        overwrite=True,
    ) as w:
        f = w.file
        f.attrs["source_aug_image_h5_path"] = "/tmp/x_image_aug.h5"
        f.attrs["source_aug_image_h5_sha256"] = "abc"
        f.attrs["aug_config_sha256"] = "deadbeef"
        f.attrs["variants_json"] = json.dumps(["v1", "v2", "v3", "v4"])
        ids_dset = w.create_1d(manifest.get("ids"), n=n_rows)
        ids_dset[:] = np.asarray([f"PAT{i:03d}" for i in range(n_rows)], dtype=object)
        srci = w.create_1d(manifest.get("source_row_index"), n=n_rows)
        srci[:] = np.arange(n_rows, dtype=np.int32)
        var = w.create_1d(manifest.get("variants"), n=n_rows)
        var[:] = np.asarray(["v1"] * n_rows, dtype=object)
        ap = w.create_1d(manifest.get("aug_params_json"), n=n_rows)
        ap[:] = np.asarray(["{}"] * n_rows, dtype=object)
        for slug in _MODS:
            d = w.create_stacked(
                manifest.get(f"latents/{slug}"),
                n=n_rows,
                spatial_shape=(4, 48, 56, 48),
            )
            d[:] = 0
        md = w.create_stacked(
            manifest.get("masks/tumor_latent"),
            n=n_rows,
            spatial_shape=(3, 48, 56, 48),
        )
        md[:] = 0
        # Inject a forbidden splits group.
        f.create_group("splits")
    with pytest.raises(H5ValidationError, match="splits"):
        assert_aug_latent_h5_valid(p, _COHORT, _MODS, mask_output_channels=3)

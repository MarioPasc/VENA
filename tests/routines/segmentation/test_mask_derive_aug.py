"""Tests for routines/segmentation/mask_derive_aug.

All tests use synthetic in-memory or tmp-file fixtures.  No real cohort H5s,
no segmenter checkpoints, no Picasso paths.

Coverage:
  1. PREMISE-FALSE fires on ids mismatch.
  2. PREMISE-FALSE fires on source_row_index mismatch.
  3. Engine writes masks/tumor_latent_soft with correct shape/dtype/nesting;
     schema bumped to 0.3.0.
  4. Running the engine twice is idempotent (same values, oracle untouched).
  5. Pre-existing 0.2.0 H5s still accepted by validate_aug_latent_soft_mask_group
     (schema version not treated as an error; missing-group is the only violation).
  6. Other groups (latents/t1pre, masks/tumor_latent) are byte-untouched after
     a write.

pytest marker: ``segmentation``
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Geometry constants (must agree with manifest constants)
# ---------------------------------------------------------------------------

_LATENT_GRID = (48, 56, 48)
_CROP_BOX = (192, 224, 192)
_SOFT_CHANNELS = 2

# Small sphere at the crop-box centre so the SDT is fast and deterministic.
_SPHERE_CENTRE = (96, 112, 96)
_SPHERE_RADIUS = 8


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_label(
    spatial: tuple[int, int, int] = _CROP_BOX,
    centre: tuple[int, int, int] = _SPHERE_CENTRE,
    radius: int = _SPHERE_RADIUS,
) -> np.ndarray:
    """Build a BraTS-2021 integer label with a spherical lesion.

    Returns int8 shaped ``(H, W, D)`` with:
    - TC (label > 0, label != 2): sphere of ``radius`` at ``centre`` (ET=4)
    - NETC (label == 1): inner core of radius ``radius // 2``
    """
    h, w, d = spatial
    z, y, x = np.mgrid[:h, :w, :d]
    dist = np.sqrt((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2)
    label = np.zeros((h, w, d), dtype=np.int8)
    label[dist <= radius] = 4  # ET (BraTS-2021 convention)
    label[dist <= radius // 2] = 1  # NETC
    return label


def _write_synthetic_aug_image_h5(
    path: Path,
    labels: list[np.ndarray],
    ids: list[str],
    source_row_indices: list[int],
) -> None:
    """Write a minimal aug-image H5 for testing.

    Datasets: ``ids``, ``source_row_index``, ``masks/tumor``.
    Labels are assumed already at LATENT_CROP_BOX (no crop/origin needed).
    """
    h, w, d = labels[0].shape

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0.0"
        f.attrs["created_at"] = "2026-07-24T00:00:00Z"
        f.attrs["producer"] = "tests.routines.segmentation.test_mask_derive_aug"
        f.attrs["config_json"] = "{}"
        f.attrs["git_sha"] = "synthetic"
        f.attrs["cohort"] = "SYNTHETIC-AUG"
        f.attrs["domain"] = "image_aug"

        vlen_str = h5py.special_dtype(vlen=str)

        ids_ds = f.create_dataset("ids", data=np.array(ids, dtype=object), dtype=vlen_str)
        ids_ds.attrs["units"] = "dimensionless"
        ids_ds.attrs["description"] = "Synthetic aug scan IDs (non-unique: 4 variants per patient)."
        ids_ds.attrs["dtype"] = "vlen-str"
        ids_ds.attrs["leading_dim"] = "n_scans"

        src_idx = np.array(source_row_indices, dtype=np.int32)
        si_ds = f.create_dataset("source_row_index", data=src_idx)
        si_ds.attrs["units"] = "dimensionless"
        si_ds.attrs["description"] = "Row index into the clean image H5."
        si_ds.attrs["dtype"] = "int32"
        si_ds.attrs["leading_dim"] = "n_scans"

        stacked = np.stack(labels, axis=0).astype(np.int8)  # (N, H, W, D)
        tumor_ds = f.create_dataset(
            "masks/tumor",
            data=stacked,
            chunks=(1, h, w, d),
            compression="gzip",
            compression_opts=4,
        )
        tumor_ds.attrs["units"] = "dimensionless"
        tumor_ds.attrs["description"] = "BraTS-2021 integer segmentation labels (pre-cropped)."
        tumor_ds.attrs["dtype"] = "int8"
        tumor_ds.attrs["leading_dim"] = "n_scans"


def _write_synthetic_aug_latent_h5(
    path: Path,
    ids: list[str],
    source_row_indices: list[int],
    schema_version: str = "0.2.0",
) -> None:
    """Write a minimal aug-latent H5 for testing.

    Datasets: ``ids``, ``source_row_index``, ``variants``, ``aug_params_json``,
    ``latents/t1pre`` (zeros), ``masks/tumor_latent`` (zeros, 3-channel old mask).
    """
    n = len(ids)
    lat_h, lat_w, lat_d = _LATENT_GRID

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = schema_version
        f.attrs["created_at"] = "2026-07-24T00:00:00Z"
        f.attrs["producer"] = "tests.routines.segmentation.test_mask_derive_aug"
        f.attrs["config_json"] = "{}"
        f.attrs["git_sha"] = "synthetic"
        f.attrs["cohort"] = "SYNTHETIC-AUG"
        f.attrs["domain"] = "latent"
        f.attrs["split_role"] = "all"
        f.attrs["longitudinal"] = False
        f.attrs["label_system"] = "brats21"
        f.attrs["crop_box"] = str(_CROP_BOX)
        f.attrs["orientation"] = "LPS"
        # Aug-specific root attrs required by validate_aug_latent_h5.
        f.attrs["source_aug_image_h5_path"] = "/synthetic/path.h5"
        f.attrs["source_aug_image_h5_sha256"] = "deadbeef"
        f.attrs["aug_config_sha256"] = "cafebabe"
        f.attrs["variants_json"] = json.dumps(["v1", "v2", "v3"])
        # manifest_json: minimal stub; only needed for validate_h5 manifest check.
        # We avoid the full manifest round-trip here — this field is checked only
        # when the caller passes a matching H5Manifest to validate_h5.
        f.attrs["manifest_json"] = json.dumps({"schema_version": schema_version})

        vlen_str = h5py.special_dtype(vlen=str)

        ids_ds = f.create_dataset("ids", data=np.array(ids, dtype=object), dtype=vlen_str)
        ids_ds.attrs["units"] = "dimensionless"
        ids_ds.attrs["description"] = "Synthetic aug scan IDs."
        ids_ds.attrs["dtype"] = "vlen-str"
        ids_ds.attrs["leading_dim"] = "n_scans"

        src_idx = np.array(source_row_indices, dtype=np.int32)
        si_ds = f.create_dataset("source_row_index", data=src_idx)
        si_ds.attrs["units"] = "dimensionless"
        si_ds.attrs["description"] = "Row index into the clean latent H5."
        si_ds.attrs["dtype"] = "int32"
        si_ds.attrs["leading_dim"] = "n_scans"

        variants_data = np.array(["v1"] * n, dtype=object)
        var_ds = f.create_dataset("variants", data=variants_data, dtype=vlen_str)
        var_ds.attrs["units"] = "dimensionless"
        var_ds.attrs["description"] = "Augmentation variant tag per row."
        var_ds.attrs["dtype"] = "vlen-str"
        var_ds.attrs["leading_dim"] = "n_scans"

        aug_params_data = np.array([json.dumps({})] * n, dtype=object)
        ap_ds = f.create_dataset("aug_params_json", data=aug_params_data, dtype=vlen_str)
        ap_ds.attrs["units"] = "dimensionless"
        ap_ds.attrs["description"] = "JSON-encoded augmentation params per row."
        ap_ds.attrs["dtype"] = "vlen-str"
        ap_ds.attrs["leading_dim"] = "n_scans"

        # Dummy latent (all zeros)
        lat_data = np.zeros((n, 4, lat_h, lat_w, lat_d), dtype=np.float32)
        lat_ds = f.create_dataset(
            "latents/t1pre",
            data=lat_data,
            chunks=(1, 4, lat_h, lat_w, lat_d),
            compression="gzip",
            compression_opts=4,
        )
        lat_ds.attrs["units"] = "latent_au"
        lat_ds.attrs["description"] = "Synthetic MAISI latent (zeros)."
        lat_ds.attrs["dtype"] = "float32"
        lat_ds.attrs["leading_dim"] = "n_scans"

        # Old 3-channel tumor_latent mask (all zeros, sentinel for byte-identity check)
        old_mask = np.zeros((n, 3, lat_h, lat_w, lat_d), dtype=np.float32)
        om_ds = f.create_dataset(
            "masks/tumor_latent",
            data=old_mask,
            chunks=(1, 3, lat_h, lat_w, lat_d),
            compression="gzip",
            compression_opts=4,
        )
        om_ds.attrs["units"] = "dimensionless"
        om_ds.attrs["description"] = "Old 3-channel tumour mask (NETC, ED, ET)."
        om_ds.attrs["dtype"] = "float32"
        om_ds.attrs["leading_dim"] = "n_scans"


def _write_aug_corpus_registry(path: Path, image_aug_h5: Path, latent_aug_h5: Path) -> None:
    """Write a minimal aug corpus registry JSON."""
    registry = {
        "cohorts": [
            {
                "name": "SYNTHETIC-AUG",
                "image_aug_h5": str(image_aug_h5),
                "latent_aug_h5": str(latent_aug_h5),
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(registry, fh)


def _read_group_bytes(h5_path: Path, group_name: str) -> bytes:
    """Read a dataset from an H5 file and return its raw bytes."""
    with h5py.File(h5_path, "r") as f:
        return f[group_name][:].tobytes()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def derivation_cfg():
    """Default DerivationConfig (grid=(48,56,48), stride=4)."""
    from vena.segmentation.config import DerivationConfig

    return DerivationConfig()


@pytest.fixture()
def target_cfg():
    """Default TargetConfig (soft=True, sigma=3.0, euclidean_percomponent)."""
    from vena.segmentation.config import TargetConfig

    return TargetConfig()


@pytest.fixture()
def aligned_aug_h5s(tmp_path: Path):
    """Return (image_aug_h5, latent_aug_h5, corpus_json) with N=3 aligned rows.

    IDs are non-unique (same patient ID, 3 intensity-only variants v1/v2/v3)
    to mirror the real aug bank structure.
    """
    # One patient, 3 intensity variants — same ID repeated, different source_row_index.
    ids = ["SYNTHETIC-001", "SYNTHETIC-001", "SYNTHETIC-001"]
    source_row_indices = [0, 0, 0]  # all point to the same clean-cache row
    labels = [_make_synthetic_label() for _ in ids]

    image_aug_h5 = tmp_path / "synthetic_image_aug.h5"
    latent_aug_h5 = tmp_path / "synthetic_latents_aug.h5"
    corpus_json = tmp_path / "corpus_aug.json"

    _write_synthetic_aug_image_h5(image_aug_h5, labels, ids, source_row_indices)
    _write_synthetic_aug_latent_h5(latent_aug_h5, ids, source_row_indices)
    _write_aug_corpus_registry(corpus_json, image_aug_h5, latent_aug_h5)

    return image_aug_h5, latent_aug_h5, corpus_json


@pytest.fixture()
def misaligned_ids_h5s(tmp_path: Path):
    """Return (image_aug_h5, latent_aug_h5, corpus_json) where image/latent ids differ."""
    ids_img = ["SYNTHETIC-001", "SYNTHETIC-002"]
    ids_lat = ["SYNTHETIC-001", "SYNTHETIC-999"]  # mismatch at row 1
    src_idx = [0, 1]
    label = _make_synthetic_label()

    image_aug_h5 = tmp_path / "image_aug_mismatch.h5"
    latent_aug_h5 = tmp_path / "latent_aug_mismatch.h5"
    corpus_json = tmp_path / "corpus_mismatch.json"

    _write_synthetic_aug_image_h5(image_aug_h5, [label, label], ids_img, src_idx)
    _write_synthetic_aug_latent_h5(latent_aug_h5, ids_lat, src_idx)
    _write_aug_corpus_registry(corpus_json, image_aug_h5, latent_aug_h5)

    return image_aug_h5, latent_aug_h5, corpus_json


@pytest.fixture()
def misaligned_src_idx_h5s(tmp_path: Path):
    """Return H5 pair where source_row_index differs between image and latent."""
    ids = ["SYNTHETIC-001", "SYNTHETIC-001"]
    src_idx_img = [0, 0]
    src_idx_lat = [0, 99]  # mismatch at row 1

    label = _make_synthetic_label()
    image_aug_h5 = tmp_path / "image_aug_srcidx.h5"
    latent_aug_h5 = tmp_path / "latent_aug_srcidx.h5"
    corpus_json = tmp_path / "corpus_srcidx.json"

    _write_synthetic_aug_image_h5(image_aug_h5, [label, label], ids, src_idx_img)
    _write_synthetic_aug_latent_h5(latent_aug_h5, ids, src_idx_lat)
    _write_aug_corpus_registry(corpus_json, image_aug_h5, latent_aug_h5)

    return image_aug_h5, latent_aug_h5, corpus_json


# ---------------------------------------------------------------------------
# Test 1: PREMISE-FALSE on ids mismatch
# ---------------------------------------------------------------------------


def test_premise_false_ids_mismatch(misaligned_ids_h5s, tmp_path: Path) -> None:
    """Engine raises SegDerivationError with PREMISE-FALSE when ids differ."""
    from routines.segmentation.mask_derive_aug.engine import (
        MaskDeriveAugEngine,
        MaskDeriveAugRoutineConfig,
    )

    from vena.segmentation.exceptions import SegDerivationError

    _image_aug_h5, _latent_aug_h5, corpus_json = misaligned_ids_h5s

    cfg = MaskDeriveAugRoutineConfig(
        corpus_registry=corpus_json,
        artifact_dir=tmp_path / "artifacts",
    )
    engine = MaskDeriveAugEngine(cfg)

    with pytest.raises(SegDerivationError, match="PREMISE-FALSE"):
        engine.run()


# ---------------------------------------------------------------------------
# Test 2: PREMISE-FALSE on source_row_index mismatch
# ---------------------------------------------------------------------------


def test_premise_false_src_idx_mismatch(misaligned_src_idx_h5s, tmp_path: Path) -> None:
    """Engine raises SegDerivationError with PREMISE-FALSE when source_row_index differs."""
    from routines.segmentation.mask_derive_aug.engine import (
        MaskDeriveAugEngine,
        MaskDeriveAugRoutineConfig,
    )

    from vena.segmentation.exceptions import SegDerivationError

    _image_aug_h5, _latent_aug_h5, corpus_json = misaligned_src_idx_h5s

    cfg = MaskDeriveAugRoutineConfig(
        corpus_registry=corpus_json,
        artifact_dir=tmp_path / "artifacts",
    )
    engine = MaskDeriveAugEngine(cfg)

    with pytest.raises(SegDerivationError, match="PREMISE-FALSE"):
        engine.run()


# ---------------------------------------------------------------------------
# Test 3: Engine write — correct shape, dtype, nesting, schema bump
# ---------------------------------------------------------------------------


def test_engine_write_shape_dtype_nesting(aligned_aug_h5s, tmp_path: Path) -> None:
    """Engine writes masks/tumor_latent_soft (3,2,48,56,48) float32 with nesting and schema bump."""
    from routines.segmentation.mask_derive_aug.engine import (
        MaskDeriveAugEngine,
        MaskDeriveAugRoutineConfig,
    )

    from vena.data.h5.augmented.latent_domain import (
        AUG_LATENT_SCHEMA_VERSION_SOFT,
        assert_aug_latent_soft_mask_group_valid,
    )
    from vena.data.h5.latent_domain.manifest import SOFT_MASK_GROUP

    _image_aug_h5, latent_aug_h5, corpus_json = aligned_aug_h5s

    cfg = MaskDeriveAugRoutineConfig(
        corpus_registry=corpus_json,
        artifact_dir=tmp_path / "artifacts",
    )
    engine = MaskDeriveAugEngine(cfg)
    artifact_dir = engine.run()

    # Artifact directory and decision.json exist.
    assert artifact_dir.is_dir(), f"artifact dir not created: {artifact_dir}"
    assert (artifact_dir / "decision.json").exists()

    with h5py.File(latent_aug_h5, "r") as f:
        # Group present.
        assert SOFT_MASK_GROUP in f, "masks/tumor_latent_soft not written"
        dset = f[SOFT_MASK_GROUP]

        # Shape: (N=3, 2, 48, 56, 48).
        assert dset.shape == (3, _SOFT_CHANNELS, *_LATENT_GRID), f"unexpected shape {dset.shape}"
        # Dtype.
        assert dset.dtype == np.dtype("float32")

        # Nesting: TC[row 0] >= NETC[row 0] elementwise.
        first_row = dset[0]  # (2, 48, 56, 48)
        tc_ch = first_row[0]
        netc_ch = first_row[1]
        max_violation = float(np.maximum(netc_ch - tc_ch, 0.0).max())
        assert max_violation <= 1e-4, f"nesting violated: max(NETC−TC) = {max_violation:.6f}"

        # Range.
        assert float(first_row.min()) >= -1e-4, "values below 0"
        assert float(first_row.max()) <= 1.0 + 1e-4, "values above 1"

        # Schema bumped to 0.3.0.
        assert str(f.attrs["schema_version"]) == AUG_LATENT_SCHEMA_VERSION_SOFT, (
            f"schema_version not bumped: {f.attrs['schema_version']!r}"
        )
        assert "mask_source" in f.attrs, "mask_source root attr missing"

    # Group-level validator passes.
    assert_aug_latent_soft_mask_group_valid(latent_aug_h5)


# ---------------------------------------------------------------------------
# Test 4: Idempotency
# ---------------------------------------------------------------------------


def test_idempotency(aligned_aug_h5s, tmp_path: Path) -> None:
    """Second run replaces masks/tumor_latent_soft with byte-identical content."""
    from routines.segmentation.mask_derive_aug.engine import (
        MaskDeriveAugEngine,
        MaskDeriveAugRoutineConfig,
    )

    from vena.data.h5.augmented.latent_domain import assert_aug_latent_soft_mask_group_valid
    from vena.data.h5.latent_domain.manifest import SOFT_MASK_GROUP

    _image_aug_h5, latent_aug_h5, corpus_json = aligned_aug_h5s

    cfg = MaskDeriveAugRoutineConfig(
        corpus_registry=corpus_json,
        artifact_dir=tmp_path / "artifacts",
    )
    engine = MaskDeriveAugEngine(cfg)

    # First run.
    engine.run()
    with h5py.File(latent_aug_h5, "r") as f:
        data_first = f[SOFT_MASK_GROUP][:].copy()

    # Second run.
    engine.run()
    with h5py.File(latent_aug_h5, "r") as f:
        data_second = f[SOFT_MASK_GROUP][:].copy()

    # Values must be byte-identical.
    assert np.array_equal(data_first, data_second), (
        "second run produced different soft-mask values (idempotency violated)"
    )

    # Validator still passes.
    assert_aug_latent_soft_mask_group_valid(latent_aug_h5)


# ---------------------------------------------------------------------------
# Test 5: Pre-existing 0.2.0 files — validate_aug_latent_soft_mask_group accepts
#          schema version "0.2.0" and reports only missing-group, not a schema error
# ---------------------------------------------------------------------------


def test_schema_02_accepted_missing_group_only(tmp_path: Path) -> None:
    """validate_aug_latent_soft_mask_group on a 0.2.0 H5 reports missing group, not schema error."""
    from vena.data.h5.augmented.latent_domain import (
        AUG_LATENT_SCHEMA_VERSION,
        validate_aug_latent_soft_mask_group,
    )
    from vena.data.h5.latent_domain.manifest import SOFT_MASK_GROUP

    # Write a minimal H5 with schema_version="0.2.0" and NO soft mask group.
    latent_02 = tmp_path / "aug_latent_02.h5"
    _write_synthetic_aug_latent_h5(latent_02, ids=["P-001"], source_row_indices=[0])

    # Confirm schema is still 0.2.0.
    with h5py.File(latent_02, "r") as f:
        assert str(f.attrs["schema_version"]) == AUG_LATENT_SCHEMA_VERSION
        assert SOFT_MASK_GROUP not in f

    violations = validate_aug_latent_soft_mask_group(latent_02)

    # Must report a missing-group violation.
    assert any("missing" in v.lower() for v in violations), (
        f"expected a 'missing group' violation; got: {violations}"
    )
    # Must NOT report a schema-version error — 0.2.0 is a supported version.
    schema_errors = [v for v in violations if "schema_version" in v]
    assert not schema_errors, (
        f"0.2.0 schema_version incorrectly flagged as invalid: {schema_errors}"
    )


# ---------------------------------------------------------------------------
# Test 6: Other groups are byte-untouched after engine write
# ---------------------------------------------------------------------------


def test_other_groups_byte_untouched(aligned_aug_h5s, tmp_path: Path) -> None:
    """latents/t1pre and masks/tumor_latent bytes are identical before and after engine run."""
    from routines.segmentation.mask_derive_aug.engine import (
        MaskDeriveAugEngine,
        MaskDeriveAugRoutineConfig,
    )

    _image_aug_h5, latent_aug_h5, corpus_json = aligned_aug_h5s

    # Snapshot both groups before the engine touches the latent H5.
    latent_bytes_before = _read_group_bytes(latent_aug_h5, "latents/t1pre")
    old_mask_bytes_before = _read_group_bytes(latent_aug_h5, "masks/tumor_latent")

    cfg = MaskDeriveAugRoutineConfig(
        corpus_registry=corpus_json,
        artifact_dir=tmp_path / "artifacts",
    )
    engine = MaskDeriveAugEngine(cfg)
    engine.run()

    # Re-read bytes after write.
    latent_bytes_after = _read_group_bytes(latent_aug_h5, "latents/t1pre")
    old_mask_bytes_after = _read_group_bytes(latent_aug_h5, "masks/tumor_latent")

    assert latent_bytes_before == latent_bytes_after, "latents/t1pre was modified by the engine"
    assert old_mask_bytes_before == old_mask_bytes_after, (
        "masks/tumor_latent (old 3-channel) was modified by the engine"
    )


# ---------------------------------------------------------------------------
# Test 7: Import isolation
# ---------------------------------------------------------------------------


def test_import_isolation() -> None:
    """routines.segmentation.mask_derive_aug resolves inside the worktree."""
    import pathlib

    import routines
    import vena

    wt = pathlib.Path(vena.__file__).resolve().parent.parent.parent
    for mod in (vena, routines):
        p = pathlib.Path(mod.__file__).resolve()
        assert p.is_relative_to(wt), f"LEAK: {mod.__name__} -> {p}"

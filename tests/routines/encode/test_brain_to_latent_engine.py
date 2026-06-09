"""End-to-end smoke for the brain-to-latent encoder routine (2026-06-09 CHANGE 2).

Builds a tiny synthetic image H5 with a known brain mask, runs a tiny synthetic
latent H5 through the encoder, and verifies the produced ``masks/brain_latent``
matches the deterministic max-pool result.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from routines.encode.brain_to_latent.engine import (
    BrainToLatentRoutineConfig,
    BrainToLatentRoutineEngine,
    _encode_brain_mask,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthetic_image_h5(path: Path, n: int = 3) -> None:
    """Create a minimal image H5 with the only attrs/datasets the encoder needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    crop_box = [192, 224, 192]
    with h5py.File(path, "w") as f:
        f.attrs["crop_box"] = json.dumps(crop_box)
        ids = np.array([f"p{i:02d}" for i in range(n)], dtype=object)
        f.create_dataset("ids", data=ids, dtype=h5py.string_dtype())
        # Each patient has a different brain-mask coverage so the test
        # distinguishes patient identity vs row index in the cache.
        brain = np.zeros((n, 200, 230, 200), dtype=np.int8)
        for i in range(n):
            brain[i, 50 : 50 + (i + 1) * 10, 60:120, 70:120] = 1
        f.create_dataset("masks/brain", data=brain)
        # Per-scan crop origin (latent encoder requires this dataset).
        # Centre the box on the native volume.
        origin = np.zeros((n, 3), dtype=np.int32)
        for axis, (native, target) in enumerate(zip((200, 230, 200), crop_box)):
            origin[:, axis] = (native - target) // 2
        f.create_dataset("crop/origin", data=origin)


def _build_synthetic_latent_h5(path: Path, ids: list[str]) -> None:
    """Create a minimal latent H5 carrying only `ids` (no other datasets needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "ids",
            data=np.array(ids, dtype=object),
            dtype=h5py.string_dtype(),
        )


def _build_synthetic_aug_latent_h5(
    path: Path,
    ids: list[str],
    variants: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "ids",
            data=np.array(ids, dtype=object),
            dtype=h5py.string_dtype(),
        )
        f.create_dataset(
            "variants",
            data=np.array(variants, dtype=object),
            dtype=h5py.string_dtype(),
        )


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_encode_brain_mask_shape_dtype() -> None:
    img = np.zeros((192, 224, 192), dtype=np.int8)
    img[40:60, 80:100, 50:90] = 1
    out = _encode_brain_mask(img, crop_origin=(0, 0, 0), target_shape=(192, 224, 192))
    assert out.shape == (1, 48, 56, 48)
    assert out.dtype == np.int8
    # The max-pool with stride 4 must produce non-zero entries because the
    # 20×20×40 brain block exceeds a single 4×4×4 chunk.
    assert out.sum() > 0


def test_encode_brain_mask_invariant_under_constant_one() -> None:
    """All-ones image collapses to all-ones latent mask."""
    img = np.ones((192, 224, 192), dtype=np.int8)
    out = _encode_brain_mask(img, crop_origin=(0, 0, 0), target_shape=(192, 224, 192))
    assert out.dtype == np.int8
    assert (out == 1).all()


def test_encode_brain_mask_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="3-D"):
        _encode_brain_mask(
            np.zeros((10, 10), dtype=np.int8),
            crop_origin=(0, 0, 0),
            target_shape=(192, 224, 192),
        )


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


def test_engine_writes_brain_latent_to_base_h5(tmp_path: Path) -> None:
    img_path = tmp_path / "cohort_image.h5"
    lat_path = tmp_path / "cohort_latents.h5"
    _build_synthetic_image_h5(img_path, n=3)
    _build_synthetic_latent_h5(lat_path, ids=["p00", "p01", "p02"])

    cfg = BrainToLatentRoutineConfig(
        target_h5=lat_path,
        source_image_h5=img_path,
        target_aug_h5=None,
        overwrite=False,
        artifacts_root=tmp_path / "artifacts",
    )
    artifact_dir = BrainToLatentRoutineEngine(cfg).run()

    assert artifact_dir.exists()
    with h5py.File(lat_path, "r") as f:
        assert "masks/brain_latent" in f
        ds = f["masks/brain_latent"]
        assert ds.shape == (3, 1, 48, 56, 48)
        assert ds.dtype == np.int8
        # Each row must be distinct because patients have different brain coverage.
        assert not (ds[0] == ds[1]).all()
        assert not (ds[1] == ds[2]).all()

    decision = json.loads((artifact_dir / "decision.json").read_text())
    assert decision["n_rows_written_base"] == 3
    assert decision["n_v4_synthesised_ones"] == 0
    assert decision["schema_version"] == "0.1.0"


def test_engine_idempotent_on_rerun(tmp_path: Path) -> None:
    img_path = tmp_path / "img.h5"
    lat_path = tmp_path / "lat.h5"
    _build_synthetic_image_h5(img_path, n=2)
    _build_synthetic_latent_h5(lat_path, ids=["p00", "p01"])

    cfg = BrainToLatentRoutineConfig(
        target_h5=lat_path,
        source_image_h5=img_path,
        artifacts_root=tmp_path / "artifacts",
    )
    BrainToLatentRoutineEngine(cfg).run()
    with h5py.File(lat_path, "r") as f:
        original = f["masks/brain_latent"][:].copy()

    # Re-run: idempotent skip because overwrite=False and data is non-zero.
    artifact_dir = BrainToLatentRoutineEngine(cfg).run()
    decision = json.loads((artifact_dir / "decision.json").read_text())
    assert decision["n_rows_written_base"] == 0  # nothing rewritten
    with h5py.File(lat_path, "r") as f:
        np.testing.assert_array_equal(f["masks/brain_latent"][:], original)


def test_engine_handles_aug_h5_with_v4_rows(tmp_path: Path) -> None:
    img_path = tmp_path / "img.h5"
    lat_path = tmp_path / "lat.h5"
    aug_path = tmp_path / "lat_aug.h5"
    _build_synthetic_image_h5(img_path, n=2)
    _build_synthetic_latent_h5(lat_path, ids=["p00", "p01"])
    _build_synthetic_aug_latent_h5(
        aug_path,
        ids=["p00", "p00", "p01", "p01"],
        variants=["v1", "v4", "v3", "v4"],
    )

    cfg = BrainToLatentRoutineConfig(
        target_h5=lat_path,
        source_image_h5=img_path,
        target_aug_h5=aug_path,
        artifacts_root=tmp_path / "artifacts",
    )
    artifact_dir = BrainToLatentRoutineEngine(cfg).run()

    with h5py.File(aug_path, "r") as f:
        ds = f["masks/brain_latent"]
        assert ds.shape == (4, 1, 48, 56, 48)
        assert bool(ds.attrs.get("v4_brain_synthesised_ones"))
        # v4 rows (indices 1, 3) are all-ones.
        assert (ds[1] == 1).all()
        assert (ds[3] == 1).all()
        # v1/v3 rows are not all-ones (the test brain mask is sparser).
        assert not (ds[0] == 1).all()
        assert not (ds[2] == 1).all()

    decision = json.loads((artifact_dir / "decision.json").read_text())
    assert decision["n_v4_synthesised_ones"] == 2
    assert decision["n_rows_written_aug"] == 4

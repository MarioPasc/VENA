"""Tests for routines/segmentation/mask_derive.

All tests use synthetic in-memory or tmp-file fixtures.  No real cohort H5s,
no segmenter checkpoints.

pytest marker: ``segmentation``
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from vena.data.h5.latent_domain.manifest import SOFT_MASK_GROUP

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Shared geometry constants (must match manifest constants)
# ---------------------------------------------------------------------------

_LATENT_GRID = (48, 56, 48)
_CROP_BOX = (192, 224, 192)
_SOFT_CHANNELS = 2

# A small sphere placed near the centre of the crop box so the avg-pool
# centroid is deterministic and easy to verify.
_SPHERE_CENTRE = (96, 112, 96)  # roughly centre of crop box
_SPHERE_RADIUS = 8  # voxels


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_label(
    spatial: tuple[int, int, int] = _CROP_BOX,
    centre: tuple[int, int, int] = _SPHERE_CENTRE,
    radius: int = _SPHERE_RADIUS,
) -> np.ndarray:
    """Build an integer BraTS-2021 label with a spherical lesion.

    Returns an int8 array shaped ``(H, W, D)`` with:
    - WT (label>0): sphere of radius ``radius`` at ``centre``
    - NETC (label==1): smaller inner sphere of radius ``radius//2``
    """
    h, w, d = spatial
    z, y, x = np.mgrid[:h, :w, :d]
    dist = np.sqrt((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2)
    label = np.zeros((h, w, d), dtype=np.int8)
    # WT: entire sphere (ET label = 4 for BraTS-2021 convention)
    label[dist <= radius] = 4
    # NETC: inner core (label = 1)
    label[dist <= radius // 2] = 1
    return label


def _make_oracle_latent_mask(label: np.ndarray) -> np.ndarray:
    """Avg-pool the binary WT mask to latent space to produce an oracle.

    Returns float32 ``(3, 48, 56, 48)`` (3-channel oracle as in existing
    ``masks/tumor_latent`` convention: channels = NETC, ED, ET).
    """
    from torch.nn.functional import avg_pool3d

    wt_binary = (label > 0).astype(np.float32)
    t = torch.from_numpy(wt_binary).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    pooled = avg_pool3d(t, kernel_size=4, stride=4).squeeze(0).squeeze(0)  # (h,w,d)
    arr = pooled.numpy()
    # Stack 3 copies (NETC, ED, ET all = WT for this synthetic oracle).
    return np.stack([arr, arr, arr], axis=0).astype(np.float32)  # (3, 48, 56, 48)


def _write_synthetic_image_h5(
    path: Path,
    labels: list[np.ndarray],
    scan_ids: list[str],
    crop_origins: list[tuple[int, int, int]] | None = None,
) -> None:
    """Write a minimal synthetic image-domain H5 for testing.

    Groups written: ``ids``, ``masks/tumor``, ``crop/origin``.
    All volumes share the same spatial shape (first label's shape).
    """
    n = len(labels)
    h, w, d = labels[0].shape
    if crop_origins is None:
        # Zero origin: volume already at crop-box size, no pad/crop needed.
        crop_origins = [(0, 0, 0)] * n

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0.0"
        f.attrs["created_at"] = "2026-07-22T00:00:00Z"
        f.attrs["producer"] = "tests.routines.segmentation.test_mask_derive"
        f.attrs["config_json"] = "{}"
        f.attrs["git_sha"] = "synthetic"
        # Semantic attrs expected by downstream validators.
        f.attrs["cohort"] = "SYNTHETIC"
        f.attrs["domain"] = "image"
        f.attrs["split_role"] = "all"
        f.attrs["longitudinal"] = False
        f.attrs["label_system"] = "brats21"
        f.attrs["crop_box"] = str(_CROP_BOX)
        f.attrs["orientation"] = "LPS"
        f.attrs["manifest_json"] = "{}"

        # ids
        vlen_str = h5py.special_dtype(vlen=str)
        ids_ds = f.create_dataset("ids", data=np.array(scan_ids, dtype=object), dtype=vlen_str)
        ids_ds.attrs["units"] = "dimensionless"
        ids_ds.attrs["description"] = "Synthetic scan IDs."
        ids_ds.attrs["dtype"] = "vlen-str"
        ids_ds.attrs["leading_dim"] = "n_scans"

        # masks/tumor
        stacked = np.stack(labels, axis=0).astype(np.int8)  # (N, H, W, D)
        tumor_ds = f.create_dataset(
            "masks/tumor",
            data=stacked,
            chunks=(1, h, w, d),
            compression="gzip",
            compression_opts=4,
        )
        tumor_ds.attrs["units"] = "dimensionless"
        tumor_ds.attrs["description"] = "BraTS-2021 integer segmentation labels."
        tumor_ds.attrs["dtype"] = "int8"
        tumor_ds.attrs["leading_dim"] = "n_scans"

        # crop/origin
        origins = np.array(crop_origins, dtype=np.int32)  # (N, 3)
        origin_ds = f.create_dataset("crop/origin", data=origins)
        origin_ds.attrs["units"] = "voxels"
        origin_ds.attrs["description"] = "Brain-centred crop origin (H, W, D) per scan."
        origin_ds.attrs["dtype"] = "int32"
        origin_ds.attrs["leading_dim"] = "n_scans"


def _write_synthetic_latent_h5(
    path: Path,
    scan_ids: list[str],
    oracle_masks: list[np.ndarray],
) -> None:
    """Write a minimal synthetic latent-domain H5 for testing.

    Groups written: ``ids``, ``latents/t1pre`` (zeros), ``masks/tumor_latent``.
    """
    n = len(scan_ids)
    lat_h, lat_w, lat_d = _LATENT_GRID

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0.0"
        f.attrs["created_at"] = "2026-07-22T00:00:00Z"
        f.attrs["producer"] = "tests.routines.segmentation.test_mask_derive"
        f.attrs["config_json"] = "{}"
        f.attrs["manifest_json"] = "{}"
        f.attrs["git_sha"] = "synthetic"
        f.attrs["cohort"] = "SYNTHETIC"
        f.attrs["domain"] = "latent"
        f.attrs["split_role"] = "all"
        f.attrs["longitudinal"] = False
        f.attrs["label_system"] = "brats21"
        f.attrs["crop_box"] = str(_CROP_BOX)
        f.attrs["orientation"] = "LPS"
        f.attrs["vae_checkpoint_sha256"] = "synthetic"

        vlen_str = h5py.special_dtype(vlen=str)
        ids_ds = f.create_dataset("ids", data=np.array(scan_ids, dtype=object), dtype=vlen_str)
        ids_ds.attrs["units"] = "dimensionless"
        ids_ds.attrs["description"] = "Synthetic scan IDs."
        ids_ds.attrs["dtype"] = "vlen-str"
        ids_ds.attrs["leading_dim"] = "n_scans"

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
        lat_ds.attrs["description"] = "Synthetic MAISI latent."
        lat_ds.attrs["dtype"] = "float32"
        lat_ds.attrs["leading_dim"] = "n_scans"

        # Oracle masks/tumor_latent (3-channel: NETC, ED, ET)
        oracle_stack = np.stack(oracle_masks, axis=0)  # (N, 3, h, w, d)
        oracle_ds = f.create_dataset(
            "masks/tumor_latent",
            data=oracle_stack,
            chunks=(1, 3, lat_h, lat_w, lat_d),
            compression="gzip",
            compression_opts=4,
        )
        oracle_ds.attrs["units"] = "dimensionless"
        oracle_ds.attrs["description"] = "Oracle BraTS soft mask in latent space."
        oracle_ds.attrs["dtype"] = "float32"
        oracle_ds.attrs["leading_dim"] = "n_scans"


def _write_corpus_registry(path: Path, image_h5: Path, latent_h5: Path) -> None:
    """Write a minimal corpus registry JSON."""
    registry = {
        "cohorts": [
            {
                "name": "SYNTHETIC",
                "image_h5": str(image_h5),
                "latent_h5": str(latent_h5),
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(registry, fh)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_label() -> np.ndarray:
    """Integer BraTS-2021 label with a spherical lesion, shape (192,224,192)."""
    return _make_synthetic_label()


@pytest.fixture()
def derivation_cfg():
    """Default DerivationConfig (grid = (48,56,48), stride = 4)."""
    from vena.segmentation.config import DerivationConfig

    return DerivationConfig()


@pytest.fixture()
def target_cfg():
    """Default TargetConfig (soft=True, sigma=3.0, euclidean_percomponent)."""
    from vena.segmentation.config import TargetConfig

    return TargetConfig()


@pytest.fixture()
def four_patient_h5s(tmp_path: Path):
    """Return (image_h5, latent_h5, corpus_json) paths for a 4-patient fixture."""
    scan_ids = [f"SYNTHETIC-{i:03d}" for i in range(4)]
    labels = [_make_synthetic_label() for _ in scan_ids]
    oracle_masks = [_make_oracle_latent_mask(lbl) for lbl in labels]

    image_h5 = tmp_path / "synthetic_image.h5"
    latent_h5 = tmp_path / "synthetic_latent.h5"
    corpus_json = tmp_path / "corpus.json"

    _write_synthetic_image_h5(image_h5, labels, scan_ids)
    _write_synthetic_latent_h5(latent_h5, scan_ids, oracle_masks)
    _write_corpus_registry(corpus_json, image_h5, latent_h5)

    return image_h5, latent_h5, corpus_json


# ---------------------------------------------------------------------------
# Helper: read oracle bytes for idempotency comparison
# ---------------------------------------------------------------------------


def _read_oracle_bytes(latent_h5: Path) -> bytes:
    """Read raw bytes of masks/tumor_latent for byte-identity checks."""
    with h5py.File(latent_h5, "r") as f:
        data = f["masks/tumor_latent"][:]
    return data.tobytes()


# ---------------------------------------------------------------------------
# Test 1: GT derivation correctness
# ---------------------------------------------------------------------------


def test_gt_derivation_correctness(synthetic_label, derivation_cfg, target_cfg) -> None:
    """GT path produces (2, 48, 56, 48) with correct nesting and graded boundary."""
    from vena.segmentation.derivation.derive import derive_latent_soft_mask

    # crop_spec=None: label already at crop-box size (192, 224, 192)
    result = derive_latent_soft_mask(
        source="gt",
        label=synthetic_label,
        crop_spec=None,
        cfg=derivation_cfg,
        target_cfg=target_cfg,
    )

    # Shape
    assert result.shape == (2, *_LATENT_GRID), f"unexpected shape {result.shape}"
    # Dtype
    assert result.dtype == torch.float32
    # Range
    assert float(result.min()) >= 0.0 - 1e-5, "values below 0"
    assert float(result.max()) <= 1.0 + 1e-5, "values above 1"

    # Nesting: NETC ≤ WT elementwise (enforced by make_soft_targets + pooling)
    wt = result[0]
    netc = result[1]
    assert torch.all(netc <= wt + 1e-5), "nesting violation: NETC > WT at some voxel"

    # WT must be non-zero (sphere present)
    assert float(wt.max()) > 0.1, "WT channel is near-zero — sphere not detected"

    # Graded boundary: values should not be all 0 or all 1 (soft probabilities)
    wt_flat = wt.reshape(-1).numpy()
    n_mid = int(((wt_flat > 0.05) & (wt_flat < 0.95)).sum())
    assert n_mid > 0, "no intermediate boundary voxels — targets may not be soft"


# ---------------------------------------------------------------------------
# Test 2: Swap-invariance (load-bearing)
# ---------------------------------------------------------------------------


def test_swap_invariance(derivation_cfg, target_cfg) -> None:
    """GT and predicted paths produce identical shape/dtype/range and attr schema."""
    from vena.segmentation.derivation.derive import derive_latent_soft_mask
    from vena.segmentation.derivation.temperature import ClassTemperatures
    from vena.segmentation.targets.soft_targets import make_soft_targets

    label = _make_synthetic_label()

    # --- GT path ---
    gt_result = derive_latent_soft_mask(
        source="gt",
        label=label,
        crop_spec=None,
        cfg=derivation_cfg,
        target_cfg=target_cfg,
    )

    # --- Predicted path (synthetic logits at identity temperature) ---
    # Build image-space logits so the predicted path can pool them to latent res.
    # make_soft_targets returns (2, H, W, D) float32 in [0,1] at image resolution.
    # sigmoid(logit(x)) == x, so with t=1.0 the predicted path reproduces the GT.
    soft_img = make_soft_targets(label, target_cfg)  # (2, 192, 224, 192)
    eps = 1e-6
    soft_clipped = soft_img.clip(eps, 1.0 - eps)
    logits = torch.from_numpy(np.log(soft_clipped / (1.0 - soft_clipped)).astype(np.float32))

    temps = ClassTemperatures(t_wt=1.0, t_netc=1.0)
    pred_result = derive_latent_soft_mask(
        source="predicted",
        seg_prediction=logits,
        temps=temps,
        crop_spec=None,
        cfg=derivation_cfg,
    )

    # Both must have identical shape, dtype, and range.
    assert gt_result.shape == pred_result.shape, (
        f"shape mismatch: GT {gt_result.shape} vs predicted {pred_result.shape}"
    )
    assert gt_result.dtype == pred_result.dtype, (
        f"dtype mismatch: GT {gt_result.dtype} vs predicted {pred_result.dtype}"
    )
    assert float(pred_result.min()) >= 0.0 - 1e-5
    assert float(pred_result.max()) <= 1.0 + 1e-5

    # Values should be very close (identity temperature + same logits → same output).
    assert torch.allclose(gt_result, pred_result, atol=1e-4), (
        "GT and predicted outputs diverge beyond tolerance for identity temperatures"
    )


# ---------------------------------------------------------------------------
# Test 3: H5 write + validate
# ---------------------------------------------------------------------------


def test_h5_write_and_validate(four_patient_h5s, tmp_path: Path) -> None:
    """Engine writes masks/tumor_latent_soft (4,2,48,56,48), validator passes."""
    from routines.segmentation.mask_derive.engine import MaskDeriveEngine, MaskDeriveRoutineConfig

    from vena.data.h5.latent_domain.manifest import (
        LATENT_SCHEMA_VERSION_SOFT,
        SOFT_MASK_GROUP,
        assert_latent_soft_mask_group_valid,
    )

    _image_h5, latent_h5, corpus_json = four_patient_h5s

    # Snapshot oracle bytes before the engine touches the latent H5.
    oracle_before = _read_oracle_bytes(latent_h5)

    cfg = MaskDeriveRoutineConfig(
        source="gt",
        corpus_registry=corpus_json,
        artifact_dir=tmp_path / "artifacts",
    )
    engine = MaskDeriveEngine(cfg)
    artifact_dir = engine.run()

    # Artifact directory must exist.
    assert artifact_dir.is_dir(), f"artifact dir not created: {artifact_dir}"
    assert (artifact_dir / "decision.json").exists()

    # Written group must be present with correct shape.
    with h5py.File(latent_h5, "r") as f:
        assert SOFT_MASK_GROUP in f, "masks/tumor_latent_soft not written"
        dset = f[SOFT_MASK_GROUP]
        assert dset.shape == (4, 2, *_LATENT_GRID), f"unexpected shape {dset.shape}"
        assert dset.dtype == np.dtype("float32")

        # Schema must have been bumped.
        assert str(f.attrs["schema_version"]) == LATENT_SCHEMA_VERSION_SOFT, (
            f"schema_version not bumped: {f.attrs['schema_version']!r}"
        )
        assert "mask_source" in f.attrs

    # Structural validator passes.
    assert_latent_soft_mask_group_valid(latent_h5, group=SOFT_MASK_GROUP)

    # Oracle group byte-identical.
    oracle_after = _read_oracle_bytes(latent_h5)
    assert oracle_before == oracle_after, "masks/tumor_latent was modified by the engine"

    # Registration centroid check (spot case: patient 0).
    _check_centroid_registration(latent_h5)


def _check_centroid_registration(latent_h5: Path) -> None:
    """WT centroid of the derived mask ≈ oracle WT-union centroid (tolerance 3 vox)."""
    with h5py.File(latent_h5, "r") as f:
        soft = f[SOFT_MASK_GROUP][0]  # (2, 48, 56, 48)
        oracle = f["masks/tumor_latent"][0]  # (3, 48, 56, 48)

    # Derived WT binary (threshold at 0.5)
    wt_derived = (soft[0] > 0.5).astype(float)
    # Oracle WT-union: clip(sum over 3 channels, 0, 1), threshold at 0.5
    oracle_union = np.clip(oracle.sum(axis=0), 0.0, 1.0)
    wt_oracle = (oracle_union > 0.5).astype(float)

    def _centroid(binary_vol: np.ndarray) -> np.ndarray:
        idx = np.array(np.where(binary_vol))
        if idx.size == 0:
            return np.zeros(3)
        return idx.mean(axis=1)

    c_derived = _centroid(wt_derived)
    c_oracle = _centroid(wt_oracle)
    dist = float(np.linalg.norm(c_derived - c_oracle))
    assert dist < 3.0, (
        f"centroid mismatch: derived={c_derived} oracle={c_oracle} dist={dist:.2f} vox"
    )


# ---------------------------------------------------------------------------
# Test 4: Idempotency
# ---------------------------------------------------------------------------


def test_idempotency(four_patient_h5s, tmp_path: Path) -> None:
    """Second engine run replaces only the target group; oracle unchanged."""
    from routines.segmentation.mask_derive.engine import MaskDeriveEngine, MaskDeriveRoutineConfig

    from vena.data.h5.latent_domain.manifest import (
        SOFT_MASK_GROUP,
        assert_latent_soft_mask_group_valid,
    )

    _image_h5, latent_h5, corpus_json = four_patient_h5s

    cfg = MaskDeriveRoutineConfig(
        source="gt",
        corpus_registry=corpus_json,
        artifact_dir=tmp_path / "artifacts",
    )
    engine = MaskDeriveEngine(cfg)

    # First run.
    engine.run()
    with h5py.File(latent_h5, "r") as f:
        data_first = f[SOFT_MASK_GROUP][:]
    oracle_first = _read_oracle_bytes(latent_h5)

    # Second run.
    engine.run()
    with h5py.File(latent_h5, "r") as f:
        data_second = f[SOFT_MASK_GROUP][:]
    oracle_second = _read_oracle_bytes(latent_h5)

    # Soft mask group must be byte-identical across runs.
    assert np.array_equal(data_first, data_second), "second run produced different soft mask values"
    # Oracle group untouched.
    assert oracle_first == oracle_second, "oracle masks/tumor_latent changed on second run"
    # Validator still passes.
    assert_latent_soft_mask_group_valid(latent_h5, group=SOFT_MASK_GROUP)


# ---------------------------------------------------------------------------
# Test 5: Additive schema (2.0.0 H5 validates before writing)
# ---------------------------------------------------------------------------


def test_schema_additive(four_patient_h5s) -> None:
    """2.0.0 H5 (before write) does not violate validate_latent_soft_mask_group.

    After writing the group the group-level validator passes;
    the old (2.0.0) validator is not invoked and 2.0.0 files are undisturbed.
    """
    from vena.data.h5.latent_domain.manifest import (
        LATENT_SCHEMA_VERSION,
        SOFT_MASK_GROUP,
        validate_latent_soft_mask_group,
    )

    _, latent_h5, _ = four_patient_h5s

    # Before writing: schema_version is "2.0.0" and group is absent.
    with h5py.File(latent_h5, "r") as f:
        assert str(f.attrs["schema_version"]) == LATENT_SCHEMA_VERSION
        assert SOFT_MASK_GROUP not in f

    # The soft-mask validator reports a missing-group violation (not a false pass).
    violations_before = validate_latent_soft_mask_group(latent_h5)
    assert any("missing" in v for v in violations_before), (
        "expected a 'missing group' violation for an un-processed 2.0.0 H5; "
        f"got: {violations_before}"
    )


# ---------------------------------------------------------------------------
# Test 6: Predicted path raises cleanly when no segmenter is available
# ---------------------------------------------------------------------------


def test_predicted_path_raises_phase2_error(four_patient_h5s, tmp_path: Path) -> None:
    """Predicted path raises SegDerivationError with a Phase-2 message."""
    from routines.segmentation.mask_derive.engine import MaskDeriveEngine, MaskDeriveRoutineConfig

    from vena.segmentation.exceptions import SegDerivationError as _SegErr

    _, _, corpus_json = four_patient_h5s
    cfg = MaskDeriveRoutineConfig(
        source="predicted",
        corpus_registry=corpus_json,
        artifact_dir=tmp_path / "artifacts",
        segmenter_checkpoints=[Path("/nonexistent/fold_0/best.ckpt")],
    )
    engine = MaskDeriveEngine(cfg)

    with pytest.raises((FileNotFoundError, _SegErr)):
        engine.run()


# ---------------------------------------------------------------------------
# Test 7: derive_latent_soft_mask raises on bad source
# ---------------------------------------------------------------------------


def test_derive_bad_source(derivation_cfg) -> None:
    """Invalid source raises SegDerivationError."""
    from vena.segmentation.derivation.derive import derive_latent_soft_mask
    from vena.segmentation.exceptions import SegDerivationError as _SegErr

    with pytest.raises(_SegErr, match="unknown source"):
        derive_latent_soft_mask(source="oops", cfg=derivation_cfg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 8: Import isolation
# ---------------------------------------------------------------------------


def test_import_isolation() -> None:
    """vena and routines modules resolve inside the worktree."""
    import pathlib

    import routines
    import vena

    wt = pathlib.Path(vena.__file__).resolve().parent.parent.parent
    for mod in (vena, routines):
        p = pathlib.Path(mod.__file__).resolve()
        assert p.is_relative_to(wt), f"LEAK: {mod.__name__} -> {p}"

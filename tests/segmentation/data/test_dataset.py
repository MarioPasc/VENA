"""Tests for SegImageDataset, build_augmentation, and RandModalityDropout.

All tests use synthetic on-disk H5 fixtures created in a tmp directory;
no real cohort data is read.

Volume shape used throughout: ``(H, W, D) = (8, 8, 8)`` (fast, captures 3-D).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Synthetic H5 fixture
# ---------------------------------------------------------------------------


def _write_synthetic_h5(
    path: Path,
    patient_ids: list[str],
    shape: tuple[int, int, int] = (8, 8, 8),
    rng: np.random.Generator | None = None,
) -> None:
    """Create a minimal image-domain H5 file for testing.

    Layout::

        patient_ids          vlen-str  (N,)
        images/t1pre         float32   (N, H, W, D)
        images/t2            float32   (N, H, W, D)
        images/flair         float32   (N, H, W, D)
        masks/tumor          int8      (N, H, W, D)   BraTS-style labels
        masks/brain          float32   (N, H, W, D)   binary skull-strip
    """
    import h5py

    if rng is None:
        rng = np.random.default_rng(0)

    n = len(patient_ids)
    h, w, d = shape

    with h5py.File(path, "w") as hf:
        # Store patient IDs as variable-length UTF-8 strings
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset("patient_ids", data=np.array(patient_ids, dtype=object), dtype=dt)

        for mod in ("t1pre", "t2", "flair"):
            data = rng.standard_normal((n, h, w, d)).astype(np.float32)
            hf.create_dataset(f"images/{mod}", data=data)

        # Tumour label: 0 background, 1 NETC, 2 edema, 4 ET (BraTS-2021)
        # Centre voxel = ET (4), adjacent = NETC (1), rest = background/edema
        label = np.zeros((n, h, w, d), dtype=np.int8)
        # Small TC (NETC+ET) region in the centre of each scan
        cx, cy, cz = h // 2, w // 2, d // 2
        label[:, cx - 1 : cx + 1, cy - 1 : cy + 1, cz - 1 : cz + 1] = 4  # ET
        label[:, cx - 2 : cx - 1, cy - 2 : cy + 2, cz - 2 : cz + 2] = 1  # NETC
        label[:, cx + 1 : cx + 2, cy - 2 : cy + 2, cz - 2 : cz + 2] = 2  # edema
        hf.create_dataset("masks/tumor", data=label)

        # Brain mask: inner 6×6×6 cube is foreground (non-zero brain)
        brain = np.zeros((n, h, w, d), dtype=np.float32)
        brain[:, 1:-1, 1:-1, 1:-1] = 1.0
        hf.create_dataset("masks/brain", data=brain)


def _make_corpus_registry(
    tmp_dir: Path,
    h5_name: str,
    cohort_name: str = "SYNTHETIC",
) -> Path:
    """Write a minimal corpus registry JSON pointing to the synthetic H5."""
    registry = {
        "schema_version": "1.0.0",
        "name": "synthetic_test_corpus",
        "cohorts": [
            {
                "name": cohort_name,
                "pathology": "preoperative_glioma",
                "label_system": "BraTS2021",
                "role": "cv",
                "image_h5": str(tmp_dir / h5_name),
            }
        ],
    }
    reg_path = tmp_dir / "corpus_test.json"
    reg_path.write_text(json.dumps(registry))
    return reg_path


def _make_data_cfg(
    corpus_registry: Path,
    image_h5_root: Path,
    k_folds: int = 5,
    fold_seed: int = 42,
    modalities: tuple[str, ...] = ("t1pre", "t2", "flair"),
) -> MagicMock:
    """Return a minimal DataConfig mock."""
    cfg = MagicMock()
    cfg.corpus_registry = corpus_registry
    cfg.image_h5_root = image_h5_root
    cfg.k_folds = k_folds
    cfg.fold_seed = fold_seed
    cfg.modalities = modalities
    return cfg


# ---------------------------------------------------------------------------
# Shared pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_h5_dir(tmp_path: Path) -> Path:
    """Create a tmp dir with one synthetic cohort H5 and a corpus registry."""
    h5_name = "SYNTHETIC_image.h5"
    patient_ids = [f"SYN_{i:03d}" for i in range(6)]
    _write_synthetic_h5(tmp_path / h5_name, patient_ids)
    _make_corpus_registry(tmp_path, h5_name)
    return tmp_path


@pytest.fixture()
def data_cfg(synthetic_h5_dir: Path) -> MagicMock:
    reg_path = synthetic_h5_dir / "corpus_test.json"
    return _make_data_cfg(
        corpus_registry=reg_path,
        image_h5_root=synthetic_h5_dir,
    )


# ---------------------------------------------------------------------------
# SegImageDataset — basic loading
# ---------------------------------------------------------------------------


class TestSegImageDatasetLoading:
    """Test that __getitem__ returns correct shapes and types."""

    def test_len(self, data_cfg: MagicMock, synthetic_h5_dir: Path) -> None:
        from vena.segmentation.data.dataset import SegImageDataset

        ids = [f"SYN_{i:03d}" for i in range(4)]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        assert len(ds) == 4

    def test_image_shape(self, data_cfg: MagicMock) -> None:
        import torch

        from vena.segmentation.data.dataset import SegImageDataset

        ids = ["SYN_000", "SYN_001"]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        sample = ds[0]
        assert "image" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape[0] == 3  # 3 modalities
        assert sample["image"].ndim == 4  # (C, H, W, D)

    def test_target_shape(self, data_cfg: MagicMock) -> None:
        from vena.segmentation.data.dataset import SegImageDataset

        ids = ["SYN_000"]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        sample = ds[0]
        assert "target" in sample
        assert sample["target"].shape[0] == 2  # [TC, NETC]
        assert sample["target"].ndim == 4

    def test_brain_shape(self, data_cfg: MagicMock) -> None:
        from vena.segmentation.data.dataset import SegImageDataset

        ids = ["SYN_000"]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        sample = ds[0]
        assert "brain" in sample
        assert sample["brain"].shape[0] == 1  # (1,H,W,D)
        assert sample["brain"].ndim == 4

    def test_patient_id_present(self, data_cfg: MagicMock) -> None:
        from vena.segmentation.data.dataset import SegImageDataset

        ids = ["SYN_002"]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        sample = ds[0]
        assert sample["patient_id"] == "SYN_002"

    def test_unknown_id_raises(self, data_cfg: MagicMock) -> None:
        from vena.segmentation.data.dataset import SegImageDataset
        from vena.segmentation.exceptions import SegDataError

        with pytest.raises(SegDataError, match="not found in H5 index"):
            SegImageDataset(["NONEXISTENT_999"], data_cfg, augment=False)


# ---------------------------------------------------------------------------
# Z-score on brain
# ---------------------------------------------------------------------------


class TestZScoreOnBrain:
    """Brain-masked z-score: mean≈0, std≈1 over nonzero voxels; background = 0."""

    def _build_volume_with_known_brain(
        self,
        shape: tuple[int, int, int] = (16, 16, 16),
    ) -> tuple:
        """
        Returns (volume, brain_mask, expected_mean, expected_std).

        The brain region is the inner 12×12×12 cube filled with unit-normal draws.
        Background is exactly 0.  After z-scoring, brain mean≈0 and brain std≈1.
        """
        from vena.segmentation.data.dataset import _zscore_brain

        rng = np.random.default_rng(7)
        sh, sw, sd = shape
        volume = np.zeros((sh, sw, sd), dtype=np.float32)
        brain_mask = np.zeros((sh, sw, sd), dtype=np.float32)

        # Non-trivial brain with known statistics
        inner = rng.standard_normal((12, 12, 12)).astype(np.float32) * 50.0 + 300.0
        volume[2:-2, 2:-2, 2:-2] = inner
        brain_mask[2:-2, 2:-2, 2:-2] = 1.0

        zscored = _zscore_brain(volume, brain_mask)
        return zscored, brain_mask

    def test_brain_mean_approx_zero(self) -> None:
        zscored, brain_mask = self._build_volume_with_known_brain()
        brain_vals = zscored[brain_mask.astype(bool)]
        mean = float(brain_vals.mean())
        assert abs(mean) < 1e-3, f"Brain mean after z-score: {mean:.6f} (expected ≈0)"

    def test_brain_std_approx_one(self) -> None:
        zscored, brain_mask = self._build_volume_with_known_brain()
        brain_vals = zscored[brain_mask.astype(bool)]
        std = float(brain_vals.std())
        assert abs(std - 1.0) < 1e-3, f"Brain std after z-score: {std:.6f} (expected ≈1)"

    def test_background_is_zero(self) -> None:
        """Background voxels (outside brain mask) must be exactly 0."""
        zscored, brain_mask = self._build_volume_with_known_brain()
        bg = zscored[brain_mask == 0]
        assert np.all(bg == 0.0), "Background voxels are not exactly 0 after z-score"

    def test_zscore_via_dataset(self, data_cfg: MagicMock) -> None:
        """End-to-end: z-score applied by SegImageDataset.__getitem__."""
        from vena.segmentation.data.dataset import SegImageDataset

        ids = ["SYN_000"]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        sample = ds[0]
        image = sample["image"].numpy()  # (3, H, W, D)
        brain = sample["brain"].numpy().squeeze(0)  # (H, W, D)

        for ch_idx in range(image.shape[0]):
            channel = image[ch_idx]
            brain_vals = channel[brain.astype(bool)]
            if len(brain_vals) == 0:
                continue  # no foreground in this synthetic scan (degenerate — skip)
            mean = float(brain_vals.mean())
            std = float(brain_vals.std())
            assert abs(mean) < 1e-2, f"Channel {ch_idx}: brain mean = {mean:.4f} (expected ≈0)"
            assert abs(std - 1.0) < 1e-2, f"Channel {ch_idx}: brain std = {std:.4f} (expected ≈1)"


# ---------------------------------------------------------------------------
# Soft target in [0, 1] range
# ---------------------------------------------------------------------------


class TestSoftTargetRange:
    def test_target_in_unit_interval(self, data_cfg: MagicMock) -> None:
        from vena.segmentation.data.dataset import SegImageDataset

        ids = ["SYN_000", "SYN_001", "SYN_002"]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        for i in range(len(ds)):
            target = ds[i]["target"].numpy()
            assert target.min() >= 0.0, f"Target min < 0: {target.min():.6f}"
            assert target.max() <= 1.0, f"Target max > 1: {target.max():.6f}"

    def test_target_dtype_float32(self, data_cfg: MagicMock) -> None:
        import torch

        from vena.segmentation.data.dataset import SegImageDataset

        ids = ["SYN_000"]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        target = ds[0]["target"]
        assert target.dtype == torch.float32


# ---------------------------------------------------------------------------
# Custom target_fn injection (stub)
# ---------------------------------------------------------------------------


class TestTargetFnInjection:
    def test_stub_target_fn(self, data_cfg: MagicMock, synthetic_h5_dir: Path) -> None:
        """A stub target_fn is called with (label, cfg) and its output is returned."""
        from vena.segmentation.config import TargetConfig
        from vena.segmentation.data.dataset import SegImageDataset

        shape_ref: list = []

        def stub_target(label, cfg, image=None):
            shape_ref.append(label.shape)
            # Return fixed (2, h, w, d) output matching label shape
            lh, lw, ld = label.shape
            return np.full((2, lh, lw, ld), 0.5, dtype=np.float32)

        ids = ["SYN_000"]
        ds = SegImageDataset(
            ids,
            data_cfg,
            augment=False,
            target_fn=stub_target,
            target_cfg=TargetConfig(),
        )
        sample = ds[0]
        assert shape_ref, "stub_target was never called"
        target = sample["target"].numpy()
        assert np.allclose(target, 0.5), "stub_target output not forwarded correctly"


# ---------------------------------------------------------------------------
# RandModalityDropout
# ---------------------------------------------------------------------------


class TestRandModalityDropout:
    """Test the custom modality-dropout transform."""

    def test_exactly_one_dropped_when_fired(self) -> None:
        """When dropout fires, exactly ONE of {t2, flair} is zeroed."""
        from vena.segmentation.data.augment import RandModalityDropout

        dropout = RandModalityDropout(p=1.0, seed=0)  # always fires
        data = {
            "t1pre": np.ones((4, 4, 4), dtype=np.float32),
            "t2": np.ones((4, 4, 4), dtype=np.float32),
            "flair": np.ones((4, 4, 4), dtype=np.float32),
        }
        result = dropout(data)

        t1pre_sum = float(result["t1pre"].sum())
        t2_sum = float(result["t2"].sum())
        flair_sum = float(result["flair"].sum())

        assert t1pre_sum > 0, "t1pre was dropped — must never happen"
        # Exactly one of t2/flair is zero; the other is non-zero
        dropped = (t2_sum == 0) + (flair_sum == 0)
        assert dropped == 1, (
            f"Expected exactly 1 dropped modality, got t2_sum={t2_sum}, flair_sum={flair_sum}"
        )

    def test_t1pre_never_dropped(self) -> None:
        """t1pre must NEVER be zeroed regardless of seed."""
        from vena.segmentation.data.augment import RandModalityDropout

        dropout = RandModalityDropout(p=1.0)
        rng = np.random.default_rng(1234)
        for _ in range(50):
            data = {
                "t1pre": rng.standard_normal((4, 4, 4)).astype(np.float32),
                "t2": rng.standard_normal((4, 4, 4)).astype(np.float32),
                "flair": rng.standard_normal((4, 4, 4)).astype(np.float32),
            }
            result = dropout(data)
            assert not np.all(result["t1pre"] == 0), "t1pre was zeroed — forbidden"

    def test_dropout_rate_measured(self) -> None:
        """Over N=500 draws at rate p, the empirical rate should be ≈p."""
        from vena.segmentation.data.augment import RandModalityDropout

        p = 0.4
        dropout = RandModalityDropout(p=p, seed=42)
        n_trials = 500
        n_dropped = 0

        for _ in range(n_trials):
            data = {
                "t1pre": np.ones((2, 2, 2), dtype=np.float32),
                "t2": np.ones((2, 2, 2), dtype=np.float32),
                "flair": np.ones((2, 2, 2), dtype=np.float32),
            }
            result = dropout(data)
            t2_zero = np.all(result["t2"] == 0)
            flair_zero = np.all(result["flair"] == 0)
            if t2_zero or flair_zero:
                n_dropped += 1

        empirical_rate = n_dropped / n_trials
        # Allow ±0.07 tolerance (≈3σ for n_trials=500, p=0.4, σ≈0.022)
        assert abs(empirical_rate - p) < 0.07, (
            f"Dropout rate: expected ≈{p:.2f}, measured {empirical_rate:.3f}"
        )

    def test_no_dropout_when_p_zero(self) -> None:
        """At p=0, no dropout ever fires."""
        from vena.segmentation.data.augment import RandModalityDropout

        dropout = RandModalityDropout(p=0.0, seed=0)
        for _ in range(20):
            data = {
                "t1pre": np.ones((4, 4, 4), dtype=np.float32),
                "t2": np.ones((4, 4, 4), dtype=np.float32),
                "flair": np.ones((4, 4, 4), dtype=np.float32),
            }
            result = dropout(data)
            assert float(result["t2"].sum()) > 0
            assert float(result["flair"].sum()) > 0


# ---------------------------------------------------------------------------
# Soft target survives spatial transforms (no binarisation)
# ---------------------------------------------------------------------------


class TestSoftTargetThroughSpatial:
    """Soft target values must stay in [0, 1] after flip and affine transforms."""

    def test_soft_target_through_flip(self) -> None:
        """RandFlipd with bilinear mode preserves [0,1] range of soft targets."""
        from monai.transforms import RandFlipd

        rng = np.random.default_rng(0)
        target = rng.uniform(0.0, 1.0, size=(2, 8, 8, 8)).astype(np.float32)
        image = rng.standard_normal((8, 8, 8)).astype(np.float32)

        flip = RandFlipd(keys=["image", "target"], prob=1.0, spatial_axis=0)
        data = {"image": image, "target": target}
        result = flip(data)

        out_target = np.asarray(result["target"])
        assert out_target.min() >= 0.0 - 1e-6, (
            f"After flip, target min = {out_target.min():.6f} (expected ≥ 0)"
        )
        assert out_target.max() <= 1.0 + 1e-6, (
            f"After flip, target max = {out_target.max():.6f} (expected ≤ 1)"
        )

    def test_soft_target_not_binarised_after_flip(self) -> None:
        """After flip, soft targets must contain intermediate float values (not 0/1 only)."""
        from monai.transforms import RandFlipd

        rng = np.random.default_rng(1)
        target = rng.uniform(0.1, 0.9, size=(2, 8, 8, 8)).astype(np.float32)
        image = rng.standard_normal((8, 8, 8)).astype(np.float32)

        flip = RandFlipd(keys=["image", "target"], prob=1.0, spatial_axis=1)
        result = flip({"image": image, "target": target})

        out_target = np.asarray(result["target"])
        # If binarised, all values would be 0 or 1; check intermediate values exist
        intermediate = ((out_target > 0.05) & (out_target < 0.95)).any()
        assert intermediate, "Target appears binarised after flip (no intermediate values)"

    def test_soft_target_through_affine_bilinear(self) -> None:
        """RandAffined with bilinear mode keeps soft targets in [0,1].

        MONAI infers spatial dims from ``ndim - 1`` for each key.  For 3D
        data both keys must be 4D (C, H, W, D) so that the grid is built
        for 3D spatial dims and is consistent across keys.  A channel-less
        (H, W, D) image with a (2, H, W, D) target would generate a 2D grid
        for the image and a 3D grid for the target, causing a grid_sample
        dimension mismatch at runtime.
        """
        from monai.transforms import RandAffined

        rng = np.random.default_rng(2)
        # Smoothly varying target to avoid numerical issues
        target = rng.uniform(0.2, 0.8, size=(2, 8, 8, 8)).astype(np.float32)
        # Use (1, H, W, D) so MONAI sees 3D spatial dims — consistent with target
        image = rng.standard_normal((1, 8, 8, 8)).astype(np.float32)

        affine = RandAffined(
            keys=["image", "target"],
            prob=1.0,
            rotate_range=(0.1,),
            mode=["bilinear", "bilinear"],
            padding_mode=["zeros", "zeros"],
        )
        result = affine({"image": image, "target": target})

        out_target = np.asarray(result["target"])
        # Bilinear interpolation can produce small extrapolation artefacts;
        # clamp to [−ε, 1+ε] is the physical tolerance
        assert out_target.min() >= -1e-4, (
            f"Affine target min = {out_target.min():.6f} (should be ≥ 0)"
        )
        assert out_target.max() <= 1.0 + 1e-4, (
            f"Affine target max = {out_target.max():.6f} (should be ≤ 1)"
        )


# ---------------------------------------------------------------------------
# build_augmentation — pipeline smoke
# ---------------------------------------------------------------------------


class TestBuildAugmentation:
    def test_returns_callable(self, data_cfg: MagicMock) -> None:
        from vena.segmentation.data.augment import build_augmentation

        pipeline = build_augmentation(data_cfg)
        assert callable(pipeline)

    def test_pipeline_runs_without_error(self, data_cfg: MagicMock) -> None:
        """The full pipeline must not raise on valid input."""
        from vena.segmentation.data.augment import build_augmentation

        rng = np.random.default_rng(5)
        pipeline = build_augmentation(data_cfg, modality_dropout_p=0.0)
        data = {
            "t1pre": rng.standard_normal((8, 8, 8)).astype(np.float32),
            "t2": rng.standard_normal((8, 8, 8)).astype(np.float32),
            "flair": rng.standard_normal((8, 8, 8)).astype(np.float32),
            "target": rng.uniform(0, 1, (2, 8, 8, 8)).astype(np.float32),
            "brain": (rng.random((8, 8, 8)) > 0.3).astype(np.float32),
        }
        result = pipeline(data)
        assert "t1pre" in result
        assert "target" in result

    def test_modality_dropout_p_controls_rate(self, data_cfg: MagicMock) -> None:
        """dropout_p=1.0 always zeroes one of {t2, flair}."""
        from vena.segmentation.data.augment import build_augmentation

        rng = np.random.default_rng(6)
        pipeline = build_augmentation(data_cfg, modality_dropout_p=1.0)

        n_dropped = 0
        n_trials = 20
        for _ in range(n_trials):
            data = {
                "t1pre": rng.standard_normal((4, 4, 4)).astype(np.float32),
                "t2": np.ones((4, 4, 4), dtype=np.float32),
                "flair": np.ones((4, 4, 4), dtype=np.float32),
                "target": rng.uniform(0, 1, (2, 4, 4, 4)).astype(np.float32),
                "brain": np.ones((4, 4, 4), dtype=np.float32),
            }
            result = pipeline(data)
            t2_zero = np.all(np.asarray(result["t2"]) == 0)
            flair_zero = np.all(np.asarray(result["flair"]) == 0)
            if t2_zero or flair_zero:
                n_dropped += 1

        # With p=1.0, every sample should have exactly one dropout
        assert n_dropped == n_trials, (
            f"Expected {n_trials} dropped samples at p=1.0, got {n_dropped}"
        )

    def test_t1c_absent_from_dataset_output(self, data_cfg: MagicMock) -> None:
        """t1c must not appear in the dataset output (label leakage)."""
        from vena.segmentation.data.dataset import SegImageDataset

        ids = ["SYN_000"]
        ds = SegImageDataset(ids, data_cfg, augment=False)
        sample = ds[0]
        assert "t1c" not in sample, "t1c must NOT be present in dataset output"


# ---------------------------------------------------------------------------
# Bug regression tests (Bug 1 + Bug 2 in _build_id_index)
# ---------------------------------------------------------------------------


def _write_schema2_h5(
    path: Path,
    scan_ids: list[str],
    patient_keys: list[str],
    offsets: list[int],
    shape: tuple[int, int, int] = (4, 4, 4),
    rng: np.random.Generator | None = None,
) -> None:
    """Write an H5 with the schema-2.0.0 layout: ``ids`` (not ``patient_ids``).

    This is the layout of the real UCSF-PDGM / BraTS H5 files produced after
    the 2026-05-19 schema bump.  The pre-fix ``_build_id_index`` would silently
    skip every cohort because it looked for ``patient_ids`` and emitted a
    WARNING — causing ``SegImageDataset`` to raise on every production run.
    """
    import h5py

    if rng is None:
        rng = np.random.default_rng(42)

    n = len(scan_ids)
    h, w, d = shape
    dt = h5py.special_dtype(vlen=str)

    with h5py.File(path, "w") as hf:
        # Schema 2.0.0: scan-level 'ids', NOT 'patient_ids'
        hf.create_dataset("ids", data=np.array(scan_ids, dtype=object), dtype=dt)
        hf.create_dataset("patients/keys", data=np.array(patient_keys, dtype=object), dtype=dt)
        hf.create_dataset("patients/offsets", data=np.array(offsets, dtype=np.int32))

        for mod in ("t1pre", "t2", "flair"):
            hf.create_dataset(
                f"images/{mod}", data=rng.standard_normal((n, h, w, d)).astype(np.float32)
            )
        label = np.zeros((n, h, w, d), dtype=np.int8)
        cx, cy, cz = h // 2, w // 2, d // 2
        label[:, cx - 1 : cx + 1, cy - 1 : cy + 1, cz - 1 : cz + 1] = 4
        hf.create_dataset("masks/tumor", data=label)
        brain = np.zeros((n, h, w, d), dtype=np.float32)
        brain[:, 1:-1, 1:-1, 1:-1] = 1.0
        hf.create_dataset("masks/brain", data=brain)


class TestBuildIdIndexBugRegressions:
    """Regression tests for Bug 1 (wrong key name) and Bug 2 (path resolution)."""

    def test_bug1_ids_key_preferred_over_patient_ids(self, tmp_path: Path) -> None:
        """Bug 1: _build_id_index must read 'ids', not 'patient_ids'.

        Pre-fix behaviour: every production H5 was silently skipped (WARNING
        logged, empty index returned), causing SegImageDataset to raise
        'N patient IDs not found in H5 index' on every real training run.
        """
        from vena.segmentation.data.dataset import _build_id_index

        scan_ids = ["SCAN_A", "SCAN_B", "SCAN_C"]
        patient_keys = scan_ids  # 1:1 for single-session
        offsets = list(range(len(scan_ids) + 1))
        h5_path = tmp_path / "SCHEMA2_image.h5"
        _write_schema2_h5(h5_path, scan_ids, patient_keys, offsets)

        registry = {
            "schema_version": "1.0.0",
            "name": "test_corpus",
            "cohorts": [
                {
                    "name": "SCHEMA2",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        }
        reg_path = tmp_path / "corpus.json"
        reg_path.write_text(json.dumps(registry))

        index = _build_id_index(reg_path, tmp_path)

        # All scan IDs must be indexed — pre-fix would return {}
        assert set(index.keys()) == set(scan_ids), (
            f"Expected {set(scan_ids)}, got {set(index.keys())}. "
            "Bug 1 not fixed: _build_id_index did not read 'ids' key."
        )

    def test_bug1_falls_back_to_patient_ids(self, tmp_path: Path) -> None:
        """Legacy 'patient_ids' key still works as a fallback (schema <2.0.0)."""
        import h5py

        from vena.segmentation.data.dataset import _build_id_index

        h5_path = tmp_path / "LEGACY_image.h5"
        legacy_ids = ["OLD_000", "OLD_001"]
        rng = np.random.default_rng(7)
        dt = h5py.special_dtype(vlen=str)

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("patient_ids", data=np.array(legacy_ids, dtype=object), dtype=dt)
            for mod in ("t1pre", "t2", "flair"):
                hf.create_dataset(
                    f"images/{mod}",
                    data=rng.standard_normal((2, 4, 4, 4)).astype(np.float32),
                )
            hf.create_dataset("masks/tumor", data=np.zeros((2, 4, 4, 4), dtype=np.int8))
            hf.create_dataset("masks/brain", data=np.ones((2, 4, 4, 4), dtype=np.float32))

        registry = {
            "schema_version": "1.0.0",
            "name": "test_corpus",
            "cohorts": [
                {
                    "name": "LEGACY",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        }
        reg_path = tmp_path / "corpus.json"
        reg_path.write_text(json.dumps(registry))

        index = _build_id_index(reg_path, tmp_path)
        assert set(index.keys()) == set(legacy_ids)

    def test_bug2_absolute_path_resolution(self, tmp_path: Path) -> None:
        """Bug 2: _build_id_index must try the registry's absolute path first.

        Pre-fix behaviour: ``image_h5_root / basename`` was always used,
        which silently produced an empty index for any cohort whose H5 lives
        in a nested subdirectory (BraTS-GLI, UPENN-GBM, etc.).
        """
        from vena.segmentation.data.dataset import _build_id_index

        # H5 lives in a nested subdir — NOT directly in image_h5_root
        nested_dir = tmp_path / "NESTED" / "subdir" / "h5"
        nested_dir.mkdir(parents=True)
        h5_path = nested_dir / "COH_image.h5"

        scan_ids = ["NSCAN_000", "NSCAN_001"]
        patient_keys = scan_ids
        offsets = list(range(len(scan_ids) + 1))
        _write_schema2_h5(h5_path, scan_ids, patient_keys, offsets)

        registry = {
            "schema_version": "1.0.0",
            "name": "test_corpus",
            "cohorts": [
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    # Absolute path pointing to nested location
                    "image_h5": str(h5_path),
                }
            ],
        }
        reg_path = tmp_path / "corpus.json"
        reg_path.write_text(json.dumps(registry))

        # image_h5_root = tmp_path (does NOT contain COH_image.h5 directly)
        index = _build_id_index(reg_path, tmp_path)

        assert set(index.keys()) == set(scan_ids), (
            "Bug 2 not fixed: _build_id_index did not resolve the nested "
            "absolute path from the registry."
        )

    def test_bug2_fallback_to_image_h5_root(self, tmp_path: Path) -> None:
        """image_h5_root / filename fallback works when absolute path is absent."""
        from vena.segmentation.data.dataset import _build_id_index

        # H5 lives directly in image_h5_root (flat layout for this test)
        h5_path = tmp_path / "COH_image.h5"
        scan_ids = ["FLAT_000", "FLAT_001"]
        patient_keys = scan_ids
        offsets = list(range(len(scan_ids) + 1))
        _write_schema2_h5(h5_path, scan_ids, patient_keys, offsets)

        registry = {
            "schema_version": "1.0.0",
            "name": "test_corpus",
            "cohorts": [
                {
                    "name": "COH",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    # Absolute path that doesn't exist (different machine path)
                    "image_h5": "/nonexistent/machine/path/COH_image.h5",
                }
            ],
        }
        reg_path = tmp_path / "corpus.json"
        reg_path.write_text(json.dumps(registry))

        # Fallback: image_h5_root / "COH_image.h5" exists → succeeds
        index = _build_id_index(reg_path, tmp_path)
        assert set(index.keys()) == set(scan_ids)

    def test_bug1_and_bug2_together(self, tmp_path: Path) -> None:
        """Combined: nested absolute path + 'ids' key (production scenario).

        This is the exact failure mode hit on every local run before the fix:
        BraTS-GLI at /nested/path/BraTS_GLI_image.h5 with schema-2.0.0 'ids'.
        """
        from vena.segmentation.data.dataset import _build_id_index

        nested_dir = tmp_path / "BRATS_GLI" / "PRE_OPERATIVE" / "h5"
        nested_dir.mkdir(parents=True)
        h5_path = nested_dir / "BraTS_GLI_image.h5"

        scan_ids = [f"BraTS-GLI-{i:05d}" for i in range(4)]
        patient_keys = scan_ids
        offsets = list(range(len(scan_ids) + 1))
        _write_schema2_h5(h5_path, scan_ids, patient_keys, offsets)

        registry = {
            "schema_version": "1.0.0",
            "name": "test_corpus",
            "cohorts": [
                {
                    "name": "BraTS-GLI",
                    "pathology": "preoperative_glioma",
                    "label_system": "BraTS2021",
                    "role": "cv",
                    "image_h5": str(h5_path),
                }
            ],
        }
        reg_path = tmp_path / "corpus.json"
        reg_path.write_text(json.dumps(registry))

        index = _build_id_index(reg_path, tmp_path)
        assert set(index.keys()) == set(scan_ids)

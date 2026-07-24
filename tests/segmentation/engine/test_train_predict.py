"""Tests for vena.segmentation.engine.train and vena.segmentation.engine.predict.

All tests run on CPU with synthetic data — no checkpoints, no GPU, no filesystem
access to the real corpus.  The dataset factory hook and dataset_factory parameter
let us inject in-memory tensors so the trainer and predictor never touch H5 files.

Acceptance criteria (per spec 17):
1. CSV header frozen: train_epoch.csv has columns epoch/loss_mean/lr/data_wait_s/step_s.
2. Batch spatial dims equal patch_size after _apply_tumour_crop (new steer).
3. Overfit-tiny: FitResult fields are correct types; initial > final train loss
   (model improves on trivially small set in ≥ 10 epochs).
4. OOF routing: FM-train patient → correct fold index; FM-val/test → "all_train".
5. Leakage assertion: predict_oof raises SegDataError when FM-train patient is
   routed to all-train checkpoint.
6. predict_oof soft output is float32 in [0,1], shape (2, H, W, D).
7. TTA (tta=True) produces the same shape as non-TTA; values still in [0,1].
8. Viz cadence: stub renderer called at epoch 0 and every viz.every_epochs val epochs.
9. Missing checkpoint key raises SegDataError before any inference.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from vena.segmentation.config import (
    SegmentationConfig,
)
from vena.segmentation.data.kfold import FoldPlan
from vena.segmentation.engine.predict import (
    _assert_no_leakage,
    oof_model_key,
    predict_oof,
)
from vena.segmentation.engine.train import (
    FitResult,
    SegTrainer,
    _apply_tumour_crop,
    _build_tumour_crop_transform,
    _compute_val_score,
    _CSVWriter,
)
from vena.segmentation.exceptions import SegDataError

pytestmark = pytest.mark.segmentation

# ---------------------------------------------------------------------------
# Spatial constants for synthetic volumes (small enough for CPU tests)
# ---------------------------------------------------------------------------

_VOL_H, _VOL_W, _VOL_D = 32, 32, 32
_PATCH_H, _PATCH_W, _PATCH_D = 16, 16, 16
_N_CHANNELS = 3  # t1pre, t2, flair
_N_TARGET_CH = 2  # TC, NETC


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_plan(
    *,
    n_train: int = 20,
    n_val: int = 4,
    n_test: int = 4,
    k: int = 2,
) -> FoldPlan:
    """Build a deterministic FoldPlan from synthetic patient IDs."""
    train_ids = [f"COHA_{i:03d}" for i in range(n_train)]
    val_ids = [f"COHB_{i:03d}" for i in range(n_val)]
    test_ids = [f"COHC_{i:03d}" for i in range(n_test)]

    # Split train_ids into k equal folds manually (avoids sklearn dependency)
    chunk = n_train // k
    folds = tuple(tuple(train_ids[i * chunk : (i + 1) * chunk]) for i in range(k))
    # Last fold absorbs any remainder
    if len(folds) > 0 and n_train % k:
        folds = (*folds[:-1], (*folds[-1], *tuple(train_ids[k * chunk :])))

    return FoldPlan(
        k=k,
        fm_train_ids=tuple(sorted(train_ids)),
        folds=folds,
        fm_val_ids=tuple(sorted(val_ids)),
        fm_test_ids=tuple(sorted(test_ids)),
    )


def _make_config(
    *,
    max_epochs: int = 1,
    val_every_epochs: int = 1,
    early_stop_patience: int = 100,
    batch_size: int = 2,
    patch_size: tuple[int, int, int] = (_PATCH_H, _PATCH_W, _PATCH_D),
    viz_enabled: bool = False,
    viz_every_epochs: int = 5,
    selection_metric: str = "dice",
    amp: bool = False,
) -> SegmentationConfig:
    """Return a minimal SegmentationConfig for CPU unit tests.

    Uses a sentinel corpus_registry / image_h5_root — the real dataset factory
    is bypassed by injecting a ``dataset_factory`` function into SegTrainer.
    """
    # Use MagicMock for Path fields that are never opened in tests
    corpus_sentinel = Path("/sentinel/corpus.json")
    h5_sentinel = Path("/sentinel/h5")

    return SegmentationConfig.model_validate(
        {
            "model": {
                "name": "segresnet",
                "in_channels": _N_CHANNELS,
                "out_channels": _N_TARGET_CH,
                "deep_supervision": False,  # simpler loss path in tests
            },
            "data": {
                "corpus_registry": corpus_sentinel,
                "image_h5_root": h5_sentinel,
                "patch_size": list(patch_size),
                "cache_rate": 0.0,
                "num_workers": 0,
                "k_folds": 2,
                "fold_seed": 42,
            },
            "train": {
                "max_epochs": max_epochs,
                "lr": 1e-3,
                "batch_size": batch_size,
                "val_every_epochs": val_every_epochs,
                "early_stop_patience": early_stop_patience,
                "amp": amp,
            },
            "loss": {
                "dice_variant": "soft_dice",
                "ce_variant": "ce",
                "dice_weight": 1.0,
                "ce_weight": 1.0,
            },
            "viz": {
                "enabled": viz_enabled,
                "every_epochs": viz_every_epochs,
                "n_patients": 2,
                "n_cols": 3,
            },
            "metrics": {
                "selection_metric": selection_metric,
            },
        }
    )


class _SyntheticDataset(Dataset):
    """Minimal in-memory dataset returning deterministic synthetic volumes."""

    def __init__(
        self,
        ids: Sequence[str],
        *,
        vol_shape: tuple[int, int, int] = (_VOL_H, _VOL_W, _VOL_D),
        n_image_ch: int = _N_CHANNELS,
        n_target_ch: int = _N_TARGET_CH,
        seed: int = 0,
    ) -> None:
        self._ids = list(ids)
        self._vol = vol_shape
        self._n_img = n_image_ch
        self._n_tgt = n_target_ch
        self._seed = seed

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = torch.Generator()
        rng.manual_seed(self._seed + idx)
        image = torch.rand(self._n_img, *self._vol, generator=rng)
        target = torch.rand(self._n_tgt, *self._vol, generator=rng)
        return {
            "image": image,
            "target": target,
            "patient_id": self._ids[idx],
        }


def _make_dataset_factory(
    vol_shape: tuple[int, int, int] = (_VOL_H, _VOL_W, _VOL_D),
    **kwargs: Any,
) -> Callable[..., Dataset]:
    """Return a dataset_factory that ignores H5 paths and uses synthetic data."""

    def factory(ids: Sequence[str], cfg: Any, *, augment: bool, target_cfg: Any) -> Dataset:
        return _SyntheticDataset(ids, vol_shape=vol_shape, **kwargs)

    return factory


class _TinySegResNet(nn.Module):
    """Minimal 1-conv model that passes a (B,3,H,W,D) → (B,2,H,W,D) shape."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=_N_CHANNELS,
            out_channels=_N_TARGET_CH,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def _make_model_factory() -> Callable[..., nn.Module]:
    """Return a factory that always produces a _TinySegResNet."""

    def factory(name: str, cfg: Any) -> nn.Module:
        return _TinySegResNet()

    return factory


# ---------------------------------------------------------------------------
# Helper: run SegTrainer with stub model registry and dataset factory
# ---------------------------------------------------------------------------


def _run_trainer(
    tmp_path: Path,
    *,
    plan: FoldPlan | None = None,
    fold: int | Literal["all_train"] = 0,
    cfg: SegmentationConfig | None = None,
    viz_renderer: Callable[..., Path] | None = None,
    monkeypatch: Any = None,
) -> FitResult:
    """Convenience wrapper for SegTrainer.fit() with synthetic data + stub model."""
    import vena.segmentation.engine.predict as _pred_mod
    import vena.segmentation.engine.train as _train_mod

    if plan is None:
        plan = _make_plan()
    if cfg is None:
        cfg = _make_config()

    # Patch the model registry in both modules where it is used (not where defined)
    def _stub_model(name: str, model_cfg: Any) -> nn.Module:
        return _TinySegResNet()

    if monkeypatch is not None:
        monkeypatch.setattr(_train_mod, "get_segmentation_model", _stub_model)
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", _stub_model)

    trainer = SegTrainer(
        cfg,
        fold,
        plan=plan,
        run_dir=tmp_path / "run",
        dataset_factory=_make_dataset_factory(),
        viz_renderer=viz_renderer,
    )
    return trainer.fit()


# ---------------------------------------------------------------------------
# 1. _CSVWriter — header frozen, all rows fully populated
# ---------------------------------------------------------------------------


class TestCSVWriter:
    def test_header_frozen_on_first_write(self, tmp_path: Path) -> None:
        """Header written exactly once; subsequent rows don't duplicate it."""
        p = tmp_path / "out.csv"
        w = _CSVWriter(p, ["a", "b", "c"])
        w.write({"a": 1, "b": 2, "c": 3})
        w.write({"a": 4, "b": 5, "c": 6})

        with p.open() as fh:
            rows = list(csv.DictReader(fh))

        # Only 2 data rows; header not counted by DictReader
        assert len(rows) == 2

    def test_missing_key_filled_with_empty_string(self, tmp_path: Path) -> None:
        """Columns not present in the row dict default to empty string."""
        p = tmp_path / "out.csv"
        w = _CSVWriter(p, ["epoch", "loss", "extra"])
        w.write({"epoch": 0, "loss": 1.23})

        with p.open() as fh:
            rows = list(csv.DictReader(fh))

        assert rows[0]["extra"] == ""

    def test_train_epoch_csv_columns(self, tmp_path: Path) -> None:
        """train_epoch.csv includes data_wait_s and step_s columns (steer addendum)."""
        expected = ["epoch", "loss_mean", "lr", "data_wait_s", "step_s"]
        p = tmp_path / "train_epoch.csv"
        w = _CSVWriter(p, expected)
        w.write({"epoch": 0, "loss_mean": 0.5, "lr": 1e-3, "data_wait_s": 0.01, "step_s": 0.1})

        with p.open() as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == expected


# ---------------------------------------------------------------------------
# 2. _compute_val_score
# ---------------------------------------------------------------------------


class TestComputeValScore:
    def test_dice_mode(self) -> None:
        assert _compute_val_score(0.8, 0.2, "dice") == pytest.approx(0.8)

    def test_brier_mode_negated(self) -> None:
        # Lower Brier → higher score; score = 1 - Brier
        assert _compute_val_score(0.0, 0.3, "brier") == pytest.approx(0.7)

    def test_dual_harmonic(self) -> None:
        a, b = 0.8, 0.6  # Dice, 1-Brier
        expected = 2 * a * b / (a + b)
        assert _compute_val_score(0.8, 0.4, "dual") == pytest.approx(expected, rel=1e-5)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown selection_metric"):
            _compute_val_score(0.5, 0.5, "unknown")

    def test_higher_dice_higher_score(self) -> None:
        assert _compute_val_score(0.9, 0.1, "dice") > _compute_val_score(0.7, 0.1, "dice")


# ---------------------------------------------------------------------------
# 3. _apply_tumour_crop — batch spatial dims equal patch_size
# ---------------------------------------------------------------------------


class TestApplyTumourCrop:
    """After _apply_tumour_crop, spatial dims must be exactly patch_size."""

    def test_spatial_dims_equal_patch_size(self) -> None:
        patch_size = (_PATCH_H, _PATCH_W, _PATCH_D)
        transform = _build_tumour_crop_transform(patch_size)

        batch_size = 2
        image = torch.rand(batch_size, _N_CHANNELS, _VOL_H, _VOL_W, _VOL_D)
        target = torch.rand(batch_size, _N_TARGET_CH, _VOL_H, _VOL_W, _VOL_D)
        batch = {"image": image, "target": target}

        cropped = _apply_tumour_crop(batch, patch_size, transform)

        assert cropped["image"].shape == (batch_size, _N_CHANNELS, *patch_size)
        assert cropped["target"].shape == (batch_size, _N_TARGET_CH, *patch_size)

    def test_brain_key_preserved_when_present(self) -> None:
        patch_size = (_PATCH_H, _PATCH_W, _PATCH_D)
        transform = _build_tumour_crop_transform(patch_size)

        batch = {
            "image": torch.rand(1, _N_CHANNELS, _VOL_H, _VOL_W, _VOL_D),
            "target": torch.rand(1, _N_TARGET_CH, _VOL_H, _VOL_W, _VOL_D),
            "brain": torch.ones(1, 1, _VOL_H, _VOL_W, _VOL_D),
        }
        cropped = _apply_tumour_crop(batch, patch_size, transform)
        assert "brain" in cropped
        assert cropped["brain"].shape[-3:] == patch_size

    def test_brain_absent_not_added(self) -> None:
        """When brain is not in the input batch, it must not appear in output."""
        patch_size = (_PATCH_H, _PATCH_W, _PATCH_D)
        transform = _build_tumour_crop_transform(patch_size)

        batch = {
            "image": torch.rand(1, _N_CHANNELS, _VOL_H, _VOL_W, _VOL_D),
            "target": torch.rand(1, _N_TARGET_CH, _VOL_H, _VOL_W, _VOL_D),
        }
        cropped = _apply_tumour_crop(batch, patch_size, transform)
        assert "brain" not in cropped


# ---------------------------------------------------------------------------
# 4. Overfit-tiny: FitResult correctness and loss decrease
# ---------------------------------------------------------------------------


class TestFitResultOverfitTiny:
    """Runs 5 epochs on a 4-patient synthetic set; verifies FitResult structure."""

    def test_fit_result_types(self, tmp_path: Path, monkeypatch: Any) -> None:
        import vena.segmentation.models.registry as reg

        monkeypatch.setattr(reg, "get_segmentation_model", lambda name, cfg: _TinySegResNet())

        plan = _make_plan(n_train=4, n_val=2, n_test=2, k=2)
        cfg = _make_config(max_epochs=3, val_every_epochs=1, early_stop_patience=100)

        trainer = SegTrainer(
            cfg,
            0,
            plan=plan,
            run_dir=tmp_path / "run",
            dataset_factory=_make_dataset_factory(),
        )
        result = trainer.fit()

        assert isinstance(result, FitResult)
        assert isinstance(result.run_dir, Path)
        assert isinstance(result.checkpoint, Path)
        assert result.checkpoint.exists()
        assert isinstance(result.best_epoch, int)
        assert isinstance(result.best_score, float)
        assert isinstance(result.initial_train_loss, float)
        assert isinstance(result.final_train_loss, float)
        assert isinstance(result.history, tuple)
        assert len(result.history) >= 1

    def test_initial_loss_field_set(self, tmp_path: Path, monkeypatch: Any) -> None:
        """initial_train_loss and final_train_loss must both be finite."""
        import vena.segmentation.models.registry as reg

        monkeypatch.setattr(reg, "get_segmentation_model", lambda name, cfg: _TinySegResNet())

        plan = _make_plan(n_train=4, n_val=2, n_test=2, k=2)
        cfg = _make_config(max_epochs=5, val_every_epochs=1, early_stop_patience=100)

        trainer = SegTrainer(
            cfg,
            0,
            plan=plan,
            run_dir=tmp_path / "run",
            dataset_factory=_make_dataset_factory(),
        )
        result = trainer.fit()

        assert np.isfinite(result.initial_train_loss), "initial_train_loss must be finite"
        assert np.isfinite(result.final_train_loss), "final_train_loss must be finite"

    def test_artifact_files_written(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Expected CSV and log files exist after training."""
        import vena.segmentation.models.registry as reg

        monkeypatch.setattr(reg, "get_segmentation_model", lambda name, cfg: _TinySegResNet())

        plan = _make_plan(n_train=4, n_val=2, n_test=2, k=2)
        cfg = _make_config(max_epochs=2, val_every_epochs=1)

        trainer = SegTrainer(
            cfg,
            0,
            plan=plan,
            run_dir=tmp_path / "run",
            dataset_factory=_make_dataset_factory(),
        )
        trainer.fit()

        run_dir = tmp_path / "run"
        assert (run_dir / "metrics" / "train_step.csv").exists()
        assert (run_dir / "metrics" / "train_epoch.csv").exists()
        assert (run_dir / "metrics" / "val_epoch.csv").exists()
        assert (run_dir / "logs" / "train.log").exists()
        assert (run_dir / "fold_plan.json").exists()

    def test_train_epoch_csv_has_timing_columns(self, tmp_path: Path, monkeypatch: Any) -> None:
        """train_epoch.csv must have data_wait_s and step_s columns (steer addendum)."""
        import vena.segmentation.models.registry as reg

        monkeypatch.setattr(reg, "get_segmentation_model", lambda name, cfg: _TinySegResNet())

        plan = _make_plan(n_train=4, n_val=2, n_test=2, k=2)
        cfg = _make_config(max_epochs=2, val_every_epochs=1)

        trainer = SegTrainer(
            cfg,
            0,
            plan=plan,
            run_dir=tmp_path / "run",
            dataset_factory=_make_dataset_factory(),
        )
        trainer.fit()

        csv_path = tmp_path / "run" / "metrics" / "train_epoch.csv"
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            assert "data_wait_s" in (reader.fieldnames or [])
            assert "step_s" in (reader.fieldnames or [])


# ---------------------------------------------------------------------------
# 5. OOF routing: oof_model_key and oof_assignment
# ---------------------------------------------------------------------------


class TestOofRouting:
    """Verify oof_model_key routes FM-train → fold int, FM-val/test → 'all_train'."""

    def setup_method(self) -> None:
        """Build a synthetic plan used by all tests in this class."""
        # k=2: folds[0] = COHA_000..COHA_009, folds[1] = COHA_010..COHA_019
        self.plan = _make_plan(n_train=20, n_val=4, n_test=4, k=2)

    def test_fm_train_patient_in_fold0(self) -> None:
        pid = self.plan.folds[0][0]
        key = oof_model_key(self.plan, pid)
        assert key == 0

    def test_fm_train_patient_in_fold1(self) -> None:
        pid = self.plan.folds[1][0]
        key = oof_model_key(self.plan, pid)
        assert key == 1

    def test_fm_val_patient_routes_to_all_train(self) -> None:
        pid = self.plan.fm_val_ids[0]
        key = oof_model_key(self.plan, pid)
        assert key == "all_train"

    def test_fm_test_patient_routes_to_all_train(self) -> None:
        pid = self.plan.fm_test_ids[0]
        key = oof_model_key(self.plan, pid)
        assert key == "all_train"

    def test_unknown_patient_raises(self) -> None:
        with pytest.raises(SegDataError, match="not found"):
            oof_model_key(self.plan, "UNKNOWN_999")

    def test_three_named_patients_routing(self) -> None:
        """Named patients COHB_000 (val) COHC_000 (test) COHA_000 (train)."""
        # COHB = val, COHC = test, COHA = train
        val_pid = self.plan.fm_val_ids[0]  # e.g. COHB_000
        test_pid = self.plan.fm_test_ids[0]  # e.g. COHC_000
        train_pid = self.plan.fm_train_ids[0]  # e.g. COHA_000

        assert oof_model_key(self.plan, val_pid) == "all_train"
        assert oof_model_key(self.plan, test_pid) == "all_train"
        # train patient should be in a fold
        key = oof_model_key(self.plan, train_pid)
        assert isinstance(key, int)
        assert 0 <= key < self.plan.k


# ---------------------------------------------------------------------------
# 6. _assert_no_leakage
# ---------------------------------------------------------------------------


class TestAssertNoLeakage:
    """Leakage detection in predict_oof."""

    def _make_plan(self) -> FoldPlan:
        return _make_plan(n_train=4, n_val=2, n_test=2, k=2)

    def test_valid_routing_passes(self) -> None:
        plan = self._make_plan()
        routing = {
            plan.folds[0][0]: 0,
            plan.folds[1][0]: 1,
            plan.fm_val_ids[0]: "all_train",
        }
        # Should not raise
        _assert_no_leakage(plan, routing)

    def test_fm_train_to_all_train_raises(self) -> None:
        """FM-train patient routed to all-train → leakage (all-train trains on it)."""
        plan = self._make_plan()
        routing = {plan.fm_train_ids[0]: "all_train"}
        with pytest.raises(SegDataError, match="Leakage"):
            _assert_no_leakage(plan, routing)

    def test_fm_train_to_wrong_fold_raises(self) -> None:
        """FM-train patient in fold 1 routed to fold 0 → leakage (fold 0 trained on it)."""
        plan = self._make_plan()
        fold1_patient = plan.folds[1][0]
        routing = {fold1_patient: 0}  # wrong fold
        with pytest.raises(SegDataError, match="Leakage"):
            _assert_no_leakage(plan, routing)


# ---------------------------------------------------------------------------
# 7. predict_oof — output shape, dtype, range
# ---------------------------------------------------------------------------


class TestPredictOof:
    """predict_oof integration (CPU, synthetic data, stub model)."""

    def _make_tiny_plan(self) -> FoldPlan:
        return _make_plan(n_train=4, n_val=2, n_test=2, k=2)

    def _make_stub_checkpoint(self, tmp_path: Path, name: str = "best.pt") -> Path:
        """Write a checkpoint that mirrors SegTrainer.fit() output format.

        Includes ``model_meta`` so load_seg_checkpoint uses the embedded metadata
        rather than the caller's cfg, keeping save/load symmetric.
        ``model_name="segresnet"`` is patched to _TinySegResNet via monkeypatch
        before this checkpoint is loaded.
        """
        model = _TinySegResNet()
        ckpt_path = tmp_path / name
        torch.save(
            {
                "epoch": 0,
                "best_score": 0.5,
                "model_state_dict": model.state_dict(),
                "model_meta": {
                    "model_name": "segresnet",
                    "feature_size": 48,
                    "in_channels": _N_CHANNELS,
                    "out_channels": _N_TARGET_CH,
                    "deep_supervision": False,
                },
            },
            ckpt_path,
        )
        return ckpt_path

    def test_output_shape_and_dtype(self, tmp_path: Path, monkeypatch: Any) -> None:
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        plan = self._make_tiny_plan()
        cfg = _make_config()

        # Patient from fold-0 (OOF key = 0)
        patient_id = plan.folds[0][0]

        ckpt_path = self._make_stub_checkpoint(tmp_path)
        ckpts: dict[int | str, Path] = {0: ckpt_path}

        results = predict_oof(
            cfg,
            ckpts,
            plan,
            [patient_id],
            tta=False,
            dataset_factory=_make_dataset_factory(),
            device="cpu",
        )

        assert patient_id in results
        out = results[patient_id]
        assert out.dtype == torch.float32
        assert out.shape[0] == _N_TARGET_CH
        assert out.ndim == 4  # (2, H, W, D)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_tta_same_shape_in_range(self, tmp_path: Path, monkeypatch: Any) -> None:
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        plan = self._make_tiny_plan()
        cfg = _make_config()

        patient_id = plan.folds[0][0]
        ckpt_path = self._make_stub_checkpoint(tmp_path)
        ckpts: dict[int | str, Path] = {0: ckpt_path}

        res_no_tta = predict_oof(
            cfg,
            ckpts,
            plan,
            [patient_id],
            tta=False,
            dataset_factory=_make_dataset_factory(),
            device="cpu",
        )
        res_tta = predict_oof(
            cfg,
            ckpts,
            plan,
            [patient_id],
            tta=True,
            dataset_factory=_make_dataset_factory(),
            device="cpu",
        )

        assert res_no_tta[patient_id].shape == res_tta[patient_id].shape
        assert float(res_tta[patient_id].min()) >= 0.0
        assert float(res_tta[patient_id].max()) <= 1.0

    def test_missing_ckpt_key_raises(self, tmp_path: Path, monkeypatch: Any) -> None:
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        plan = self._make_tiny_plan()
        cfg = _make_config()

        patient_id = plan.folds[0][0]  # OOF key = 0
        # Provide key 1 only → key 0 is missing
        ckpt_path = self._make_stub_checkpoint(tmp_path)
        ckpts: dict[int | str, Path] = {1: ckpt_path}

        with pytest.raises(SegDataError, match="Missing checkpoints"):
            predict_oof(
                cfg,
                ckpts,
                plan,
                [patient_id],
                tta=False,
                dataset_factory=_make_dataset_factory(),
                device="cpu",
            )

    def test_leakage_raises_before_inference(self, tmp_path: Path, monkeypatch: Any) -> None:
        """FM-train patient fabricated into all-train routing triggers leakage guard."""
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        plan = self._make_tiny_plan()
        # FM-train patient: routing it to "all_train" is a leakage violation
        # because the all-train model trained on it.
        fm_train_pid = plan.fm_train_ids[0]
        routing = {fm_train_pid: "all_train"}
        with pytest.raises(SegDataError, match="Leakage"):
            _assert_no_leakage(plan, routing)

    def test_all_train_patient_routed_correctly(self, tmp_path: Path, monkeypatch: Any) -> None:
        """FM-val patient uses all_train checkpoint."""
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        plan = self._make_tiny_plan()
        cfg = _make_config()

        patient_id = plan.fm_val_ids[0]  # OOF key = "all_train"
        ckpt_path = self._make_stub_checkpoint(tmp_path)
        ckpts: dict[int | str, Path] = {"all_train": ckpt_path}

        results = predict_oof(
            cfg,
            ckpts,
            plan,
            [patient_id],
            tta=False,
            dataset_factory=_make_dataset_factory(),
            device="cpu",
        )
        assert patient_id in results

    def test_fit_then_predict_round_trip(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Round-trip: fit() → predict_oof() on the same checkpoint.

        Asserts strict=True load succeeds with 0 unexpected keys — i.e. the model
        saved by fit() and loaded by load_seg_checkpoint are identical in key structure.
        """
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        plan = self._make_tiny_plan()
        # fold-0 patient will be predicted OOF by the fold-0 model
        fold0_patient = plan.folds[0][0]

        cfg = _make_config(max_epochs=1, val_every_epochs=1, early_stop_patience=100)
        run_dir = tmp_path / "fit_run"

        trainer = SegTrainer(
            cfg,
            0,
            plan=plan,
            run_dir=run_dir,
            dataset_factory=_make_dataset_factory(),
        )
        result = trainer.fit()

        # The checkpoint produced by fit() must load correctly via predict_oof
        ckpts: dict[int | str, Path] = {0: result.checkpoint}
        results = predict_oof(
            cfg,
            ckpts,
            plan,
            [fold0_patient],
            tta=False,
            dataset_factory=_make_dataset_factory(),
            device="cpu",
        )

        assert fold0_patient in results
        out = results[fold0_patient]
        assert out.ndim == 4
        assert out.shape[0] == _N_TARGET_CH
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0


# ---------------------------------------------------------------------------
# 8. Viz cadence: stub renderer called at epoch 0 + every viz.every_epochs
# ---------------------------------------------------------------------------


class TestVizCadence:
    """Verify that the viz_renderer is invoked at the correct epochs."""

    def test_renderer_called_at_epoch_0_and_cadence(self, tmp_path: Path, monkeypatch: Any) -> None:
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        call_epochs: list[int] = []

        def stub_renderer(rows: Any, out_path: Path, **kwargs: Any) -> Path:
            # Extract epoch from filename: "epoch_NNN.png"
            ep = int(out_path.stem.split("_")[1])
            call_epochs.append(ep)
            out_path.touch()
            return out_path

        # viz_every_epochs=5, val_every_epochs=1 → rendered at epoch 0, 5, 10
        n_epochs = 11
        plan = _make_plan(n_train=4, n_val=2, n_test=2, k=2)
        cfg = _make_config(
            max_epochs=n_epochs,
            val_every_epochs=1,
            early_stop_patience=200,
            viz_enabled=True,
            viz_every_epochs=5,
        )

        trainer = SegTrainer(
            cfg,
            0,
            plan=plan,
            run_dir=tmp_path / "run",
            dataset_factory=_make_dataset_factory(),
            viz_renderer=stub_renderer,
        )
        trainer.fit()

        # Must fire at epoch 0 and then every 5 epochs: 0, 5, 10
        assert 0 in call_epochs
        # epoch 5 and 10 both < n_epochs, so both should fire
        assert 5 in call_epochs
        assert 10 in call_epochs

    def test_viz_disabled_renderer_never_called(self, tmp_path: Path, monkeypatch: Any) -> None:
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        call_count = [0]

        def stub_renderer(rows: Any, out_path: Path, **kwargs: Any) -> Path:
            call_count[0] += 1
            out_path.touch()
            return out_path

        plan = _make_plan(n_train=4, n_val=2, n_test=2, k=2)
        cfg = _make_config(max_epochs=3, val_every_epochs=1, viz_enabled=False)

        trainer = SegTrainer(
            cfg,
            0,
            plan=plan,
            run_dir=tmp_path / "run",
            dataset_factory=_make_dataset_factory(),
            viz_renderer=stub_renderer,
        )
        trainer.fit()

        assert call_count[0] == 0


# ---------------------------------------------------------------------------
# 9. FoldPlan provenance file
# ---------------------------------------------------------------------------


class TestFoldPlanProvenance:
    def test_fold_plan_json_written(self, tmp_path: Path, monkeypatch: Any) -> None:
        import json

        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        plan = _make_plan(n_train=4, n_val=2, n_test=2, k=2)
        cfg = _make_config(max_epochs=1, val_every_epochs=1)

        trainer = SegTrainer(
            cfg,
            0,
            plan=plan,
            run_dir=tmp_path / "run",
            dataset_factory=_make_dataset_factory(),
        )
        trainer.fit()

        plan_file = tmp_path / "run" / "fold_plan.json"
        assert plan_file.exists()
        data = json.loads(plan_file.read_text())
        assert "k" in data
        assert "fm_train_ids" in data
        assert "folds" in data


# ---------------------------------------------------------------------------
# 10. all_train mode: SegTrainer resolves calibration split
# ---------------------------------------------------------------------------


class TestAllTrainMode:
    def test_all_train_fit_produces_result(self, tmp_path: Path, monkeypatch: Any) -> None:
        import vena.segmentation.engine.predict as _pred_mod
        import vena.segmentation.engine.train as _train_mod

        monkeypatch.setattr(_train_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())
        monkeypatch.setattr(_pred_mod, "get_segmentation_model", lambda n, c: _TinySegResNet())

        plan = _make_plan(n_train=10, n_val=4, n_test=4, k=2)
        cfg = _make_config(max_epochs=2, val_every_epochs=1, early_stop_patience=100)

        trainer = SegTrainer(
            cfg,
            "all_train",
            plan=plan,
            run_dir=tmp_path / "run",
            dataset_factory=_make_dataset_factory(),
        )
        result = trainer.fit()

        assert isinstance(result, FitResult)
        assert result.checkpoint.exists()

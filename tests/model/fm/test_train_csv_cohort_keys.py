"""Tests for the per-cohort key discovery in TrainMetricsCSV.

Both ``cfm_cohort_*`` and ``contrastive_cohort_*`` keys must land in
``train_step.csv`` (every optimiser step) and ``train_epoch.csv`` (per-epoch
mean/std). Single-cohort runs leave them out of ``callback_metrics`` and the
header silently suppresses them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from vena.model.fm.lightning.callbacks.train_csv import (
    _COHORT_PREFIXES,
    _EPOCH_AGG_KEYS,
    TrainMetricsCSV,
)

pytestmark = pytest.mark.unit


def test_cohort_prefixes_cover_cfm_and_contrastive() -> None:
    assert "cfm_cohort_" in _COHORT_PREFIXES
    assert "contrastive_cohort_" in _COHORT_PREFIXES


def _make_fake_trainer(
    global_step: int,
    current_epoch: int,
    callback_metrics: dict[str, float],
    lr: float = 1e-4,
) -> SimpleNamespace:
    """Build a Lightning-trainer-shaped stub the callback can consume."""
    return SimpleNamespace(
        global_step=global_step,
        current_epoch=current_epoch,
        callback_metrics=callback_metrics,
        optimizers=[SimpleNamespace(param_groups=[{"lr": lr}])],
    )


def _make_fake_module(ema_decay: float | None = None) -> SimpleNamespace:
    if ema_decay is None:
        return SimpleNamespace(ema=None)
    return SimpleNamespace(
        ema=SimpleNamespace(get_current_decay=lambda: ema_decay),
    )


def test_train_step_csv_carries_contrastive_cohort_columns(tmp_path: Path) -> None:
    cb = TrainMetricsCSV(out_dir=tmp_path)
    metrics = {
        "train/cfm": 0.5,
        "train/contrastive": -0.2,
        "train/total": 0.3,
        "train/cfm_cohort_BraTS-GLI": 0.51,
        "train/cfm_cohort_LUMIERE": 0.49,
        "train/contrastive_cohort_BraTS-GLI": -0.18,
        "train/contrastive_cohort_LUMIERE": -0.22,
    }
    cb.on_train_batch_end(_make_fake_trainer(1, 0, metrics), _make_fake_module())
    cb.on_train_batch_end(_make_fake_trainer(2, 0, metrics), _make_fake_module())

    rows = list(csv.DictReader((tmp_path / "train_step.csv").open()))
    assert len(rows) == 2
    for row in rows:
        assert row["cfm_cohort_BraTS-GLI"] != ""
        assert row["cfm_cohort_LUMIERE"] != ""
        assert row["contrastive_cohort_BraTS-GLI"] != ""
        assert row["contrastive_cohort_LUMIERE"] != ""


def test_train_epoch_csv_carries_contrastive_cohort_columns(tmp_path: Path) -> None:
    cb = TrainMetricsCSV(out_dir=tmp_path)
    metrics = {
        "train/cfm": 0.5,
        "train/contrastive": -0.2,
        "train/total": 0.3,
        "train/cfm_cohort_BraTS-GLI": 0.51,
        "train/cfm_cohort_LUMIERE": 0.49,
        "train/contrastive_cohort_BraTS-GLI": -0.18,
        "train/contrastive_cohort_LUMIERE": -0.22,
    }
    cb.on_train_epoch_start(_make_fake_trainer(0, 0, {}), _make_fake_module())
    cb.on_train_batch_end(_make_fake_trainer(1, 0, metrics), _make_fake_module())
    cb.on_train_batch_end(_make_fake_trainer(2, 0, metrics), _make_fake_module())
    cb.on_train_epoch_end(_make_fake_trainer(2, 0, metrics), _make_fake_module())

    header = (tmp_path / "train_epoch.csv").open().readline().strip().split(",")
    for cohort in ("BraTS-GLI", "LUMIERE"):
        assert f"cfm_cohort_{cohort}_mean" in header
        assert f"cfm_cohort_{cohort}_std" in header
        assert f"contrastive_cohort_{cohort}_mean" in header
        assert f"contrastive_cohort_{cohort}_std" in header


def test_single_cohort_run_suppresses_contrastive_cohort_columns(tmp_path: Path) -> None:
    cb = TrainMetricsCSV(out_dir=tmp_path)
    # No per-cohort keys at all — single-cohort training (B sampled from one
    # cohort) skips the breakdown in the LightningModule.
    metrics = {"train/cfm": 0.5, "train/contrastive": -0.2, "train/total": 0.3}
    cb.on_train_epoch_start(_make_fake_trainer(0, 0, {}), _make_fake_module())
    cb.on_train_batch_end(_make_fake_trainer(1, 0, metrics), _make_fake_module())
    cb.on_train_epoch_end(_make_fake_trainer(1, 0, metrics), _make_fake_module())

    header = (tmp_path / "train_epoch.csv").open().readline().strip().split(",")
    assert not any("cfm_cohort_" in h for h in header)
    assert not any("contrastive_cohort_" in h for h in header)


# --------------------------------------------------------------------------
# S3 LPL columns
# --------------------------------------------------------------------------


def test_epoch_agg_keys_carry_S3_diagnostics() -> None:
    """S3 columns must be in the epoch-CSV header so train_epoch.csv has a
    stable shape across S1 / S2 / S3 runs.
    """
    for k in ("lpl", "lambda_img_active", "hi_frac", "lpl_wt", "lpl_notwt"):
        assert k in _EPOCH_AGG_KEYS, f"_EPOCH_AGG_KEYS missing required S3 key '{k}'"
    # Per-block keys for K=2 (blocks 2, 3) + K=5 (blocks 2, 5).
    for blk in (2, 3, 5):
        assert f"lpl_b{blk}" in _EPOCH_AGG_KEYS, f"missing lpl_b{blk}"


def test_S3_train_step_csv_carries_lpl_keys(tmp_path: Path) -> None:
    """``train/lpl``/``train/hi_frac``/``train/lambda_img_active`` flow into
    the first row's columns. The callback freezes the header from the first
    successful batch — these keys must be present then or they vanish.
    """
    cb = TrainMetricsCSV(out_dir=tmp_path)
    metrics = {
        "train/cfm": 0.5,
        "train/total": 0.4,
        "train/lpl": 0.07,
        "train/lambda_img_active": 0.5,
        "train/hi_frac": 0.6,
        "train/lpl_b2": 0.03,
        "train/lpl_b3": 0.04,
        "train/lpl_wt": 0.02,
        "train/lpl_notwt": 0.05,
    }
    cb.on_train_batch_end(_make_fake_trainer(1, 0, metrics), _make_fake_module())
    rows = list(csv.DictReader((tmp_path / "train_step.csv").open()))
    assert len(rows) == 1
    for k in ("lpl", "lambda_img_active", "hi_frac", "lpl_b2", "lpl_b3", "lpl_wt", "lpl_notwt"):
        assert k in rows[0], f"train_step.csv header missing '{k}'"
        assert rows[0][k] != "", f"train_step.csv column '{k}' empty"


def test_S3_train_epoch_csv_carries_lpl_means(tmp_path: Path) -> None:
    cb = TrainMetricsCSV(out_dir=tmp_path)
    metrics = {
        "train/cfm": 0.5,
        "train/total": 0.4,
        "train/lpl": 0.07,
        "train/lambda_img_active": 1.0,
        "train/hi_frac": 0.6,
    }
    cb.on_train_epoch_start(_make_fake_trainer(0, 0, {}), _make_fake_module())
    cb.on_train_batch_end(_make_fake_trainer(1, 0, metrics), _make_fake_module())
    cb.on_train_batch_end(_make_fake_trainer(2, 0, metrics), _make_fake_module())
    cb.on_train_epoch_end(_make_fake_trainer(2, 0, metrics), _make_fake_module())
    header = (tmp_path / "train_epoch.csv").open().readline().strip().split(",")
    for col in ("lpl_mean", "lpl_std", "lambda_img_active_mean", "hi_frac_mean"):
        assert col in header, f"train_epoch.csv header missing '{col}'"

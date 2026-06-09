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

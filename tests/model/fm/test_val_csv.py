"""Unit tests for ValMetricsCSV."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vena.model.fm.lightning.callbacks.val_csv import COLUMNS, ValMetricsCSV


def _write_rows(path: Path, rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)


@pytest.mark.unit
def test_val_csv_writes_header(tmp_path: Path) -> None:
    cb = ValMetricsCSV(tmp_path / "val.csv")
    with (tmp_path / "val.csv").open("r") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert tuple(header) == COLUMNS


@pytest.mark.unit
def test_val_csv_truncates_past_resumed_epoch(tmp_path: Path) -> None:
    p = tmp_path / "val.csv"
    _write_rows(
        p,
        [
            [0, 100, 5, "full", "0.1", "", "", "", "", "", "", "", "", 50, "t"],
            [3, 400, 5, "full", "0.2", "", "", "", "", "", "", "", "", 50, "t"],
            [10, 1000, 5, "full", "0.3", "", "", "", "", "", "", "", "", 50, "t"],
        ],
    )
    cb = ValMetricsCSV(p)
    trainer = MagicMock()
    trainer.current_epoch = 3
    module = MagicMock()
    cb.on_train_start(trainer, module)
    with p.open("r") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    assert [int(r[0]) for r in rows] == [0, 3]


@pytest.mark.unit
def test_val_csv_flushes_collapsed_metrics(tmp_path: Path) -> None:
    p = tmp_path / "val.csv"
    cb = ValMetricsCSV(p)
    module = MagicMock()
    module.collapse_val_metrics.return_value = {
        (5, "full"): {
            "mse_latent_mean": 0.5,
            "mse_latent_std": 0.1,
            "l1_latent_mean": 0.3,
            "l1_latent_std": 0.05,
            "cosine_latent_mean": 0.97,
            "psnr_image_mean": 28.0,
            "psnr_image_std": 1.0,
            "ssim_image_mean": 0.88,
            "ssim_image_std": 0.02,
            "n_patients": 4,
        }
    }
    trainer = MagicMock()
    trainer.current_epoch = 1
    trainer.global_step = 200
    cb.on_validation_epoch_end(trainer, module)
    with p.open("r") as f:
        rows = list(csv.reader(f))
    # header + one data row
    assert len(rows) == 2
    assert rows[1][0] == "1"
    assert rows[1][2] == "5"
    assert rows[1][3] == "full"
    # Metric cells are populated (the bug fixed: not blank).
    assert rows[1][4] == "0.5"  # mse_latent_mean
    assert rows[1][5] == "0.1"  # mse_latent_std
    assert rows[1][13] == "4"  # n_patients


@pytest.mark.unit
def test_collapse_then_write_integration(tmp_path: Path) -> None:
    """Guards the ordering bug: a RAW accumulator must collapse to populated
    cells. Mirrors the real module's collapse helper without instantiating the
    (checkpoint-loading) LightningModule."""
    from vena.model.fm.lightning.module import _agg_to_stats

    raw = {
        (5, "full"): {
            "mse": [0.4, 0.6],
            "l1": [0.25, 0.35],
            "cosine": [0.96, 0.98],
            "psnr": [27.0, 29.0],
            "ssim": [0.87, 0.89],
            "n_patients": 2,
        }
    }
    p = tmp_path / "val.csv"
    cb = ValMetricsCSV(p)
    module = MagicMock()
    module.collapse_val_metrics.return_value = {k: _agg_to_stats(v) for k, v in raw.items()}
    trainer = MagicMock()
    trainer.current_epoch = 2
    trainer.global_step = 300
    cb.on_validation_epoch_end(trainer, module)
    with p.open("r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["mse_latent_mean"] == "0.5"
    assert row["n_patients"] == "2"
    assert row["psnr_image_mean"] == "28"
    assert row["mse_latent_std"] != ""

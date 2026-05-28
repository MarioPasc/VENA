"""Unit tests for NFETimingProbe and NFETimingCSV."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vena.model.fm.inference import NFETimingProbe
from vena.model.fm.lightning.callbacks.nfe_timing import NFETimingCSV


@pytest.mark.unit
def test_timing_probe_discards_warmup() -> None:
    probe = NFETimingProbe(use_cuda_sync=False)
    for i in range(4):
        with probe.section("trunk"):
            time.sleep(0.001 * (i + 1))
    agg = probe.aggregate(drop_first=True)
    assert agg["trunk"]["n"] == 3


@pytest.mark.unit
def test_nfe_timing_csv_emits_row_per_epoch_nfe(tmp_path: Path) -> None:
    cb = NFETimingCSV(tmp_path)
    module = MagicMock()
    module.collapse_nfe_timing.return_value = [
        {
            "nfe": 5,
            "t_trunk_mean_sec": 0.12,
            "t_controlnet_mean_sec": 0.03,
            "t_decode_sec": 0.40,
            "t_total_mean_sec": 0.5,
            "t_total_std_sec": 0.01,
            "gpu_mem_peak_mb": 1024.0,
            "n_patients_measured": 4,
        },
    ]
    trainer = MagicMock()
    trainer.current_epoch = 0
    cb.on_validation_epoch_end(trainer, module)

    out = tmp_path / "nfe_timing_epoch_000.csv"
    assert out.is_file()
    with out.open("r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == "0"
    assert row["nfe"] == "5"
    # Per-component columns are now populated (was the gap fixed in this change).
    assert row["t_trunk_mean_sec"] == "0.12"
    assert row["t_controlnet_mean_sec"] == "0.03"
    assert row["t_decode_sec"] == "0.4"
    assert row["n_patients_measured"] == "4"


@pytest.mark.unit
def test_nfe_timing_csv_blanks_nan(tmp_path: Path) -> None:
    cb = NFETimingCSV(tmp_path)
    module = MagicMock()
    module.collapse_nfe_timing.return_value = [
        {"nfe": 2, "t_trunk_mean_sec": None, "t_decode_sec": float("nan")},
    ]
    trainer = MagicMock()
    trainer.current_epoch = 1
    cb.on_validation_epoch_end(trainer, module)
    with (tmp_path / "nfe_timing_epoch_001.csv").open("r") as f:
        row = list(csv.DictReader(f))[0]
    assert row["t_trunk_mean_sec"] == ""
    assert row["t_decode_sec"] == ""

"""Unit tests for NFETimingProbe and NFETimingCSV."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vena.model.fm.inference import NFETimingProbe
from vena.model.fm.lightning.callbacks.nfe_timing import COLUMNS, NFETimingCSV


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
    module._nfe_timing_buffer = [
        {"epoch": 0, "nfe": 5, "t_total_mean_sec": 0.5, "t_total_std_sec": 0.01,
         "gpu_mem_peak_mb": 1024.0, "n_patients_measured": 4},
    ]
    trainer = MagicMock()
    trainer.current_epoch = 0
    cb.on_validation_epoch_end(trainer, module)

    out = tmp_path / "nfe_timing_epoch_000.csv"
    assert out.is_file()
    with out.open("r") as f:
        rows = list(csv.reader(f))
    assert tuple(rows[0]) == COLUMNS
    assert rows[1][0] == "0" and rows[1][1] == "5"

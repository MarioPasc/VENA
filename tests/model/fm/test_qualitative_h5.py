"""Unit tests for QualitativeH5Writer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import h5py
import pytest
import torch

from vena.model.fm.lightning.callbacks.qualitative import QualitativeH5Writer


@pytest.mark.unit
def test_qualitative_writer_appends_nfes_to_same_patient(tmp_path: Path) -> None:
    cb = QualitativeH5Writer(tmp_path, run_id="run-x")
    module = MagicMock()
    module._qualitative_buffer = {
        ("pid_A", 2): torch.zeros(4, 8, 8, 8),
        ("pid_A", 5): torch.ones(4, 8, 8, 8),
        ("pid_B", 5): torch.full((4, 8, 8, 8), 0.5),
    }
    trainer = MagicMock()
    trainer.current_epoch = 1
    trainer.global_step = 25
    cb.on_validation_epoch_end(trainer, module)

    path = tmp_path / "epoch_001.h5"
    assert path.is_file()
    with h5py.File(path, "r") as f:
        assert "predictions/pid_A/nfe_2" in f
        assert "predictions/pid_A/nfe_5" in f
        assert "predictions/pid_B/nfe_5" in f
        assert f.attrs["run_id"] == "run-x"


@pytest.mark.unit
def test_qualitative_writer_empty_buffer_is_noop(tmp_path: Path) -> None:
    cb = QualitativeH5Writer(tmp_path, run_id="run-x")
    module = MagicMock()
    module._qualitative_buffer = {}
    trainer = MagicMock()
    trainer.current_epoch = 0
    cb.on_validation_epoch_end(trainer, module)
    assert list(tmp_path.glob("epoch_*.h5")) == []

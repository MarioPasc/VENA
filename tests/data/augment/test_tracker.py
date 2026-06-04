"""Smoke tests for :class:`AugmentationTracker` CSV output."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from vena.data.augment.online.tracker import _CSV_NAME, AugmentationTracker


class _StubTrainer:
    def __init__(self, current_epoch: int = 0) -> None:
        self.current_epoch = int(current_epoch)


def test_tracker_records_combos(tmp_path: Path) -> None:
    tracker = AugmentationTracker(out_dir=tmp_path)
    tracker.on_train_batch_end(
        _StubTrainer(0),
        SimpleNamespace(),
        outputs=None,
        batch={"_aug_combo": ["flip_lr", "none"]},
        batch_idx=0,
    )
    tracker.on_train_batch_end(
        _StubTrainer(0),
        SimpleNamespace(),
        outputs=None,
        batch={"_aug_combo": ["flip_lr"]},
        batch_idx=1,
    )
    tracker.on_train_epoch_end(_StubTrainer(0), SimpleNamespace())
    tracker.on_train_batch_end(
        _StubTrainer(1),
        SimpleNamespace(),
        outputs=None,
        batch={"_aug_combo": ["flip_lr+translate_hp4"]},
        batch_idx=0,
    )
    tracker.on_fit_end(_StubTrainer(1), SimpleNamespace())

    path = tmp_path / _CSV_NAME
    assert path.is_file()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert {(r["epoch"], r["combo"], r["count"]) for r in rows} == {
        ("0", "flip_lr", "2"),
        ("0", "none", "1"),
        ("1", "flip_lr+translate_hp4", "1"),
    }


def test_tracker_no_aug_no_csv(tmp_path: Path) -> None:
    """Training without augmentation must produce no CSV."""
    tracker = AugmentationTracker(out_dir=tmp_path)
    tracker.on_train_batch_end(
        _StubTrainer(0),
        SimpleNamespace(),
        outputs=None,
        batch={"z_t1c": object()},
        batch_idx=0,
    )
    tracker.on_fit_end(_StubTrainer(0), SimpleNamespace())
    assert not (tmp_path / _CSV_NAME).exists()


def test_tracker_handles_string_batch_key(tmp_path: Path) -> None:
    """``batch["_aug_combo"]`` may be a single str if batch_size == 1."""
    tracker = AugmentationTracker(out_dir=tmp_path)
    tracker.on_train_batch_end(
        _StubTrainer(0),
        SimpleNamespace(),
        outputs=None,
        batch={"_aug_combo": "flip_lr"},
        batch_idx=0,
    )
    tracker.on_fit_end(_StubTrainer(0), SimpleNamespace())
    with (tmp_path / _CSV_NAME).open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"epoch": "0", "combo": "flip_lr", "count": "1"}]

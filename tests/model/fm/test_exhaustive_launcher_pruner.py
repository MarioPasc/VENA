"""Unit tests for the ExhaustiveValLauncher snapshot pruner (P1.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vena.model.fm.lightning.callbacks.exhaustive_launcher import ExhaustiveValLauncher

pytestmark = pytest.mark.unit


def _make_launcher(tmp_path: Path, keep: int) -> ExhaustiveValLauncher:
    """Build a launcher without spawning anything — bypass __init__'s sys.executable."""
    launcher = ExhaustiveValLauncher.__new__(ExhaustiveValLauncher)
    launcher.run_dir = tmp_path
    launcher.run_id = "test"
    launcher.job_base = {}
    launcher.every_epochs = 1
    launcher.device = "cuda:1"
    launcher.block_until_complete = False
    launcher.prune_snapshots_keep = keep
    launcher.cwd = tmp_path
    launcher.python = "python"
    launcher.out_root = tmp_path / "exhaustive_val"
    launcher.out_root.mkdir(parents=True, exist_ok=True)
    launcher.gpu_log = launcher.out_root / "gpu_usage.log"
    launcher._proc = None
    launcher._proc_epoch = None
    return launcher


def _populate_epoch(out_root: Path, epoch: int, with_trunk: bool = True) -> Path:
    """Create one epoch dir with all the standard exhaustive-val files."""
    d = out_root / f"epoch_{epoch:03d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ema_snapshot.pt").write_bytes(b"x" * 1024)
    if with_trunk:
        (d / "trunk_ema_snapshot.pt").write_bytes(b"x" * 2048)
    (d / "metrics.csv").write_text("cohort,patient_id\n")
    (d / "latent_preds.h5").write_bytes(b"hdf5")
    return d


def test_pruner_keeps_k_most_recent_snapshots(tmp_path: Path) -> None:
    launcher = _make_launcher(tmp_path, keep=2)
    for e in (0, 1, 2, 3, 4):
        _populate_epoch(launcher.out_root, e)

    launcher._prune_old_snapshots(current_epoch=4)

    # Epochs 3 and 4 keep their snapshots; 0/1/2 lose them.
    for e in (0, 1, 2):
        d = launcher.out_root / f"epoch_{e:03d}"
        assert not (d / "ema_snapshot.pt").exists(), f"epoch {e} ema should be pruned"
        assert not (d / "trunk_ema_snapshot.pt").exists(), f"epoch {e} trunk should be pruned"
    for e in (3, 4):
        d = launcher.out_root / f"epoch_{e:03d}"
        assert (d / "ema_snapshot.pt").exists()
        assert (d / "trunk_ema_snapshot.pt").exists()


def test_pruner_never_touches_metrics_or_h5(tmp_path: Path) -> None:
    launcher = _make_launcher(tmp_path, keep=1)
    for e in (0, 1, 2):
        _populate_epoch(launcher.out_root, e)
    launcher._prune_old_snapshots(current_epoch=2)
    for e in (0, 1, 2):
        d = launcher.out_root / f"epoch_{e:03d}"
        assert (d / "metrics.csv").exists()
        assert (d / "latent_preds.h5").exists()


def test_pruner_disabled_when_keep_zero(tmp_path: Path) -> None:
    launcher = _make_launcher(tmp_path, keep=0)
    for e in (0, 1, 2, 3):
        _populate_epoch(launcher.out_root, e)
    launcher._prune_old_snapshots(current_epoch=3)
    for e in (0, 1, 2, 3):
        assert (launcher.out_root / f"epoch_{e:03d}" / "ema_snapshot.pt").exists()


def test_pruner_no_op_when_few_epochs(tmp_path: Path) -> None:
    launcher = _make_launcher(tmp_path, keep=5)
    for e in (0, 1):
        _populate_epoch(launcher.out_root, e)
    launcher._prune_old_snapshots(current_epoch=1)
    # Fewer than keep dirs — nothing pruned.
    assert (launcher.out_root / "epoch_000" / "ema_snapshot.pt").exists()
    assert (launcher.out_root / "epoch_001" / "ema_snapshot.pt").exists()


def test_pruner_handles_missing_trunk_snapshot(tmp_path: Path) -> None:
    """Frozen-trunk runs don't write trunk_ema_snapshot.pt — pruner must not error."""
    launcher = _make_launcher(tmp_path, keep=1)
    for e in (0, 1, 2):
        _populate_epoch(launcher.out_root, e, with_trunk=False)
    launcher._prune_old_snapshots(current_epoch=2)
    assert not (launcher.out_root / "epoch_000" / "ema_snapshot.pt").exists()
    assert (launcher.out_root / "epoch_002" / "ema_snapshot.pt").exists()


def test_pruner_sorts_numerically_across_thousand_boundary(tmp_path: Path) -> None:
    """Regression: launcher writes ``epoch_{epoch:03d}`` (min-width 3), so
    ``epoch_1000`` is a valid sibling of ``epoch_975``. Sorting by ``p.name``
    is lexicographic and places ``epoch_1000`` between ``epoch_100`` and
    ``epoch_125``, which made the freshly-written ≥1000 epoch dir fall
    outside the keep window and lose its just-written ``ema_snapshot.pt``
    before the spawned subprocess could load it (production runs
    ``2026-06-12_01-27-55_s1_fft_cfm_*`` and ``2026-06-12_06-32-25_s2_fft_contrastive_*``
    failed every cadence from epoch 1000 onward with FileNotFoundError).
    """
    launcher = _make_launcher(tmp_path, keep=2)
    # A realistic snapshot of the run directory at the moment epoch=1000
    # launches: every cadence from 0 to 1000 in steps of 25.
    for e in range(0, 1025, 25):
        _populate_epoch(launcher.out_root, e)

    launcher._prune_old_snapshots(current_epoch=1000)

    # The two most-recent epoch dirs (numerically) are 975 and 1000.
    assert (launcher.out_root / "epoch_975" / "ema_snapshot.pt").exists()
    assert (launcher.out_root / "epoch_1000" / "ema_snapshot.pt").exists()
    assert (launcher.out_root / "epoch_1000" / "trunk_ema_snapshot.pt").exists()
    # Every other 3-digit-padded epoch lost its snapshot, including the ones
    # that would have been "kept" under the broken lex-sort (epoch_950, _925).
    for e in range(0, 975, 25):
        d = launcher.out_root / f"epoch_{e:03d}"
        assert not (d / "ema_snapshot.pt").exists(), f"epoch {e} should be pruned"

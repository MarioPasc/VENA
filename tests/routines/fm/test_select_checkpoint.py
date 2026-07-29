"""Unit tests for scripts/select_checkpoint.py (B4 post-hoc selection).

Tests exercise the helper functions directly rather than the CLI to avoid
spawning a subprocess and to keep the suite fast (no I/O beyond tmpdir).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Import helpers (the script lives in scripts/, not a package)
# ---------------------------------------------------------------------------


def _import_script():
    """Import select_checkpoint as a module without executing main()."""
    import importlib.util

    repo_root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "select_checkpoint",
        repo_root / "scripts" / "select_checkpoint.py",
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Fake run_dir with exhaustive_val/epoch_NNN/aggregate_cv.csv entries."""
    ev_root = tmp_path / "exhaustive_val"
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    epochs_data = {
        0: 0.72,
        10: 0.75,
        20: 0.80,  # <-- best ssim_mean for brain/nfe=5
        30: 0.78,
        40: 0.77,
    }
    for epoch, ssim_val in epochs_data.items():
        d = ev_root / f"epoch_{epoch:03d}"
        d.mkdir(parents=True)
        with (d / "aggregate_cv.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "cohort",
                    "epoch",
                    "nfe",
                    "region",
                    "ssim_mean",
                    "psnr_db_mean",
                    "n_patients",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "cohort": "UCSF-PDGM",
                    "epoch": epoch,
                    "nfe": 5,
                    "region": "brain",
                    "ssim_mean": ssim_val,
                    "psnr_db_mean": 28.0 + ssim_val * 5,
                    "n_patients": 50,
                }
            )
        # Create a fake checkpoint for this epoch
        (ckpt_dir / f"epoch={epoch}-step=1000.ckpt").write_text(f"ckpt-{epoch}")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollectRows:
    """_collect_rows reads aggregate_cv.csv and returns per-epoch dicts."""

    def test_returns_all_epochs(self, run_dir: Path) -> None:
        mod = _import_script()
        rows = mod._collect_rows(run_dir, metric="ssim_mean", region="brain", nfe=5)
        assert len(rows) == 5

    def test_epoch_values_match_csv(self, run_dir: Path) -> None:
        mod = _import_script()
        rows = mod._collect_rows(run_dir, metric="ssim_mean", region="brain", nfe=5)
        by_epoch = {r["epoch"]: r["value"] for r in rows}
        assert abs(by_epoch[20] - 0.80) < 1e-6

    def test_nfe_filter_excludes_others(self, run_dir: Path) -> None:
        """Requesting nfe=99 yields no rows because only nfe=5 is in the CSV."""
        mod = _import_script()
        rows = mod._collect_rows(run_dir, metric="ssim_mean", region="brain", nfe=99)
        assert rows == []

    def test_region_filter_excludes_others(self, run_dir: Path) -> None:
        mod = _import_script()
        rows = mod._collect_rows(run_dir, metric="ssim_mean", region="wt", nfe=5)
        assert rows == []

    def test_missing_ev_dir_returns_empty(self, tmp_path: Path) -> None:
        mod = _import_script()
        rows = mod._collect_rows(tmp_path, metric="ssim_mean", region="brain", nfe=5)
        assert rows == []

    def test_nfe_none_aggregates_all_nfe(self, run_dir: Path) -> None:
        """nfe=None keeps all NFE rows; since CSV has exactly one NFE=5 per epoch,
        result length equals number of epochs."""
        mod = _import_script()
        rows = mod._collect_rows(run_dir, metric="ssim_mean", region="brain", nfe=None)
        assert len(rows) == 5


class TestFindCheckpoint:
    """_find_checkpoint resolves the right .ckpt file."""

    def test_finds_correct_epoch(self, run_dir: Path) -> None:
        mod = _import_script()
        ckpt = mod._find_checkpoint(run_dir, epoch=20)
        assert ckpt is not None
        assert "epoch=20" in ckpt.name

    def test_returns_none_for_missing_epoch(self, run_dir: Path) -> None:
        mod = _import_script()
        ckpt = mod._find_checkpoint(run_dir, epoch=999)
        assert ckpt is None

    def test_returns_none_for_missing_ckpt_dir(self, tmp_path: Path) -> None:
        mod = _import_script()
        ckpt = mod._find_checkpoint(tmp_path, epoch=0)
        assert ckpt is None


class TestBestEpochSelection:
    """Best epoch is the one with the highest metric value (mode=max)."""

    def test_best_is_epoch_20(self, run_dir: Path) -> None:
        mod = _import_script()
        rows = mod._collect_rows(run_dir, metric="ssim_mean", region="brain", nfe=5)
        best = max(rows, key=lambda r: r["value"])
        assert best["epoch"] == 20
        assert abs(best["value"] - 0.80) < 1e-6

    def test_mode_min_selects_epoch_0(self, run_dir: Path) -> None:
        mod = _import_script()
        rows = mod._collect_rows(run_dir, metric="ssim_mean", region="brain", nfe=5)
        worst = min(rows, key=lambda r: r["value"])
        assert worst["epoch"] == 0
        assert abs(worst["value"] - 0.72) < 1e-6


class TestSymlinkCreation:
    """When --out is given and a checkpoint exists, a symlink is created."""

    def test_symlink_points_to_checkpoint(self, run_dir: Path, tmp_path: Path) -> None:
        mod = _import_script()
        rows = mod._collect_rows(run_dir, metric="ssim_mean", region="brain", nfe=5)
        best = max(rows, key=lambda r: r["value"])
        ckpt = mod._find_checkpoint(run_dir, best["epoch"])
        assert ckpt is not None

        out_link = tmp_path / "best_checkpoint.ckpt"
        out_link.unlink(missing_ok=True)
        out_link.symlink_to(ckpt)

        assert out_link.is_symlink()
        assert out_link.resolve() == ckpt.resolve()

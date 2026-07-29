"""Unit tests for the B1/B2 exhaustive-val aggregation fixes.

B1: patient-mean then cohort-mean (instead of raw scan-row mean).
B2: test_only cohorts must never reach aggregate_cv.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from routines.fm.exhaustive_val.engine import ExhaustiveValEngine

pytestmark = pytest.mark.unit

_write_cv = ExhaustiveValEngine._write_aggregate_cv_csv


def _row(
    cohort: str,
    patient_id: str,
    nfe: int,
    psnr_db: float,
    ssim: float,
    role: str = "cv",
    n_scans: int = 1,
) -> dict:
    """Minimal metric row."""
    return {
        "cohort": cohort,
        "patient_id": patient_id,
        "nfe": nfe,
        "psnr_db": psnr_db,
        "ssim": ssim,
        "mae": 0.1,
        "mse": 0.01,
        "psnr_db_wt": float("nan"),
        "ssim_wt": float("nan"),
        "mae_wt": float("nan"),
        "mse_wt": float("nan"),
        "psnr_db_bg": float("nan"),
        "ssim_bg": float("nan"),
        "mae_bg": float("nan"),
        "mse_bg": float("nan"),
        "psnr_db_bnwt": float("nan"),
        "ssim_bnwt": float("nan"),
        "mae_bnwt": float("nan"),
        "mse_bnwt": float("nan"),
        "psnr_db_netc": float("nan"),
        "ssim_netc": float("nan"),
        "mae_netc": float("nan"),
        "mse_netc": float("nan"),
        "psnr_db_ed": float("nan"),
        "ssim_ed": float("nan"),
        "mae_ed": float("nan"),
        "mse_ed": float("nan"),
        "psnr_db_et": float("nan"),
        "ssim_et": float("nan"),
        "mae_et": float("nan"),
        "mse_et": float("nan"),
        "psnr_db_brain": float("nan"),
        "ssim_brain": float("nan"),
        "mae_whole": float("nan"),
        "mse_whole": float("nan"),
        "role": role,
    }


class TestAggregateCV:
    """B1: patient-mean then cohort-mean aggregation."""

    def test_patient_mean_weights_patients_equally(self, tmp_path: Path) -> None:
        """A patient with 7 scans must count as 1, same as a patient with 1 scan."""
        # Patient A: 7 scans, all psnr_db=30
        # Patient B: 1 scan, psnr_db=20
        # Scan-weighted mean = (7*30 + 20) / 8 = 27.5
        # Patient-mean-then-cohort-mean = (30 + 20) / 2 = 25.0
        rows = [_row("cohort1", "A", nfe=5, psnr_db=30.0, ssim=0.9) for _ in range(7)]
        rows += [_row("cohort1", "B", nfe=5, psnr_db=20.0, ssim=0.8)]

        out = tmp_path / "agg_cv.csv"
        _write_cv(out, rows)

        with out.open() as f:
            reader = csv.DictReader(f)
            data = [r for r in reader if r["region"] == "whole" and int(r["nfe"]) == 5]

        assert len(data) == 1
        psnr_mean = float(data[0]["psnr_db_mean"])
        assert abs(psnr_mean - 25.0) < 0.01, f"expected patient-balanced 25.0, got {psnr_mean}"

    def test_single_patient_single_scan(self, tmp_path: Path) -> None:
        rows = [_row("cohort1", "P1", nfe=2, psnr_db=32.5, ssim=0.92)]
        out = tmp_path / "agg_cv.csv"
        _write_cv(out, rows)

        with out.open() as f:
            data = [r for r in csv.DictReader(f) if r["region"] == "whole"]
        assert len(data) == 1
        assert abs(float(data[0]["psnr_db_mean"]) - 32.5) < 0.01

    def test_multiple_cohorts_independent(self, tmp_path: Path) -> None:
        rows = [
            _row("cohort_a", "P1", nfe=5, psnr_db=30.0, ssim=0.9),
            _row("cohort_b", "P2", nfe=5, psnr_db=20.0, ssim=0.8),
        ]
        out = tmp_path / "agg_cv.csv"
        _write_cv(out, rows)

        with out.open() as f:
            data = {r["cohort"]: r for r in csv.DictReader(f) if r["region"] == "whole"}

        assert abs(float(data["cohort_a"]["psnr_db_mean"]) - 30.0) < 0.01
        assert abs(float(data["cohort_b"]["psnr_db_mean"]) - 20.0) < 0.01

    def test_empty_cv_rows_writes_empty_file(self, tmp_path: Path) -> None:
        out = tmp_path / "agg_cv.csv"
        _write_cv(out, [])
        assert out.exists()
        assert out.stat().st_size == 0


class TestAggregateCVB2Guard:
    """B2: test_only cohorts must not reach aggregate_cv.csv."""

    def test_test_only_rows_raise_assertion_error(self, tmp_path: Path) -> None:
        rows = [_row("held_out_cohort", "P1", nfe=5, psnr_db=30.0, ssim=0.9, role="test_only")]
        out = tmp_path / "agg_cv.csv"
        with pytest.raises(AssertionError, match="test_only"):
            _write_cv(out, rows)

    def test_mixed_roles_raises_on_test_only(self, tmp_path: Path) -> None:
        rows = [
            _row("cv_cohort", "P1", nfe=5, psnr_db=30.0, ssim=0.9, role="cv"),
            _row("test_cohort", "P2", nfe=5, psnr_db=20.0, ssim=0.8, role="test_only"),
        ]
        out = tmp_path / "agg_cv.csv"
        with pytest.raises(AssertionError, match="test_only"):
            _write_cv(out, rows)

    def test_cv_only_rows_do_not_raise(self, tmp_path: Path) -> None:
        rows = [_row("cohort1", "P1", nfe=5, psnr_db=30.0, ssim=0.9, role="cv")]
        out = tmp_path / "agg_cv.csv"
        _write_cv(out, rows)  # must not raise

    def test_missing_role_defaults_to_cv(self, tmp_path: Path) -> None:
        """Rows without a 'role' key are treated as 'cv' (backward-compat)."""
        rows = [_row("cohort1", "P1", nfe=5, psnr_db=30.0, ssim=0.9)]
        # Drop the 'role' key to simulate old code paths.
        for r in rows:
            r.pop("role", None)
        out = tmp_path / "agg_cv.csv"
        _write_cv(out, rows)  # must not raise

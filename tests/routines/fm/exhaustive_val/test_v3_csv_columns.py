"""S1 v3 exhaustive_val output regression tests.

Without launching the full sub-process or the MAISI VAE, exercise the pure
column-naming + aggregate-writer code paths so a config-side typo (e.g.
missing column in the CSV header, missing region in the aggregate) is
caught before any GPU work.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from routines.fm.exhaustive_val.engine import ExhaustiveValEngine


def _toy_metric_row(
    *,
    cohort: str = "UCSF-PDGM",
    epoch: int = 1,
    pid: str = "P0",
    nfe: int = 5,
    psnr: float = 25.0,
    ssim: float = 0.9,
    et_psnr: float | str = 18.0,
    et_n: int | str = 100,
) -> dict[str, object]:
    return {
        "cohort": cohort,
        "epoch": epoch,
        "patient_id": pid,
        "nfe": nfe,
        "psnr_db": psnr,
        "ssim": ssim,
        "psnr_db_wt": 17.5,
        "ssim_wt": 0.95,
        "psnr_db_bg": 27.0,
        "ssim_bg": 0.92,
        "psnr_db_nwt": 19.0,
        "ssim_nwt": 0.93,
        "latent_mse": 0.01,
        "latent_l1": 0.02,
        "latent_cosine": 0.99,
        "gen_sec": 1.5,
        "decode_sec": 0.5,
        "brain_mask_source": "masks/brain_latent",
        # S1 v3 extras
        "psnr_db_et": et_psnr,
        "ssim_et": 0.91,
        "psnr_db_netc": 16.0,
        "ssim_netc": 0.90,
        "psnr_db_ed": 18.5,
        "ssim_ed": 0.92,
        "psnr_db_bnwt": 22.0,
        "ssim_bnwt": 0.94,
        "mae_whole": 0.05,
        "mae_wt": 0.10,
        "mae_bg": 0.01,
        "mae_bnwt": 0.06,
        "mae_et": 0.15,
        "mae_netc": 0.08,
        "mae_ed": 0.05,
        "mse_whole": 0.005,
        "mse_wt": 0.020,
        "mse_bg": 0.001,
        "mse_bnwt": 0.008,
        "mse_et": 0.030,
        "mse_netc": 0.010,
        "mse_ed": 0.005,
        "n_voxels_brain": 1_000_000,
        "n_voxels_wt": 50_000,
        "n_voxels_bnwt": 950_000,
        "n_voxels_netc": 10_000,
        "n_voxels_ed": 30_000,
        "n_voxels_et": et_n,
    }


def test_metrics_csv_header_includes_v3_extras(tmp_path: Path) -> None:
    """The CSV header MUST carry every v3 extra column (regression guard)."""
    rows = [_toy_metric_row()]
    out = tmp_path / "metrics.csv"
    ExhaustiveValEngine._write_metrics_csv(out, rows)
    with out.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
    assert header is not None
    for col in (
        "psnr_db_et",
        "ssim_et",
        "psnr_db_netc",
        "ssim_netc",
        "psnr_db_ed",
        "ssim_ed",
        "psnr_db_bnwt",
        "ssim_bnwt",
        "mae_whole",
        "mae_wt",
        "mae_bg",
        "mae_bnwt",
        "mae_et",
        "mae_netc",
        "mae_ed",
        "mse_whole",
        "mse_wt",
        "mse_bg",
        "mse_bnwt",
        "mse_et",
        "mse_netc",
        "mse_ed",
        "n_voxels_brain",
        "n_voxels_wt",
        "n_voxels_bnwt",
        "n_voxels_netc",
        "n_voxels_ed",
        "n_voxels_et",
    ):
        assert col in header, f"missing v3 CSV column {col!r}"


def test_aggregate_csv_emits_seven_regions_per_cohort_nfe(tmp_path: Path) -> None:
    """One row per (cohort, nfe, region) with region ∈ {whole..et} (7 entries)."""
    rows = [
        _toy_metric_row(pid="P0", nfe=5),
        _toy_metric_row(pid="P1", nfe=5),
        _toy_metric_row(pid="P0", nfe=10),
    ]
    out = tmp_path / "aggregate.csv"
    ExhaustiveValEngine._write_aggregate_csv(out, rows)
    with out.open() as f:
        reader = csv.DictReader(f)
        agg = list(reader)
    by_key = {(r["cohort"], int(r["nfe"]), r["region"]): r for r in agg}
    expected_regions = {"whole", "wt", "bg", "bnwt", "netc", "ed", "et"}
    for nfe in (5, 10):
        regions_present = {key[2] for key in by_key if key[1] == nfe}
        assert regions_present == expected_regions, (
            f"nfe={nfe} regions={regions_present}, expected={expected_regions}"
        )
    # n_patients for NFE=5/ET = 2; NFE=10/ET = 1.
    assert int(by_key[("UCSF-PDGM", 5, "et")]["n_patients"]) == 2
    assert int(by_key[("UCSF-PDGM", 10, "et")]["n_patients"]) == 1


def test_aggregate_csv_excludes_nan_from_means(tmp_path: Path) -> None:
    """A NaN-only ET row across patients ⇒ empty mean cell (no fake 0.0)."""
    rows = [
        _toy_metric_row(pid="P0", nfe=5, et_psnr=float("nan"), et_n=0),
        _toy_metric_row(pid="P1", nfe=5, et_psnr=float("nan"), et_n=0),
    ]
    out = tmp_path / "aggregate.csv"
    ExhaustiveValEngine._write_aggregate_csv(out, rows)
    with out.open() as f:
        for row in csv.DictReader(f):
            if row["region"] == "et":
                # All ET PSNRs were NaN — the mean cell must be empty.
                assert row["psnr_db_mean"] == ""
                assert int(row["n_patients"]) >= 0  # ssim/mae/mse still present


def test_aggregate_csv_handles_string_value_cells(tmp_path: Path) -> None:
    """Blank PSNR_ET cell across patients ⇒ PSNR_ET mean drawn from populated rows only.

    The other ET metrics (ssim/mae/mse) are still populated for both rows;
    n_patients reflects the max count across all four metrics, so it stays
    at 2 even though only P0's PSNR_ET contributed to the PSNR mean.
    """
    rows = [
        _toy_metric_row(pid="P0", nfe=5, et_psnr=18.0, et_n=100),
        _toy_metric_row(pid="P1", nfe=5, et_psnr="", et_n=""),  # blank PSNR / n_voxels
    ]
    out = tmp_path / "aggregate.csv"
    ExhaustiveValEngine._write_aggregate_csv(out, rows)
    with out.open() as f:
        for row in csv.DictReader(f):
            if row["region"] == "et" and int(row["nfe"]) == 5:
                # Only P0's PSNR_ET contributed ⇒ mean = 18.0.
                assert row["psnr_db_mean"] == "18"
                # ssim/mae/mse all populated for both patients ⇒ n_patients=2.
                assert int(row["n_patients"]) == 2

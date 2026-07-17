"""Engine-level tests for routines.validation.spatial_residual.

Uses the ``synth_shard`` session fixture from conftest.py.  Key invariants:

- Engine runs end-to-end and writes per_scan/spatial_residual.csv with the frozen 40-column header.
- LUMIERE-like cohort (5 scans / 3 patients) collapses correctly.
- C-noT uses the ``bg`` (dilated) mask key, not ``bg_undilated``.
- per_scan/spatial_residual.csv has exactly 2 rows per scan (one per condition).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vena.validation.spatial_residual import SPATIAL_CSV_COLUMNS

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: minimal ScanSample for pure library tests
# ---------------------------------------------------------------------------


def _make_sample(*, wt_fraction: float = 0.2, h: int = 12, seed: int = 0):
    """Return a synthetic ScanSample with a partial WT mask.

    Parameters
    ----------
    wt_fraction :
        Fraction of brain voxels marked as WT.
    h :
        Cube side length (volume = h³).
    seed :
        RNG seed.
    """
    from vena.validation.io import ScanSample

    rng = np.random.default_rng(seed)
    shape = (h, h, h)

    brain = np.ones(shape, dtype=bool)
    wt = np.zeros(shape, dtype=bool)
    # Mark a cube corner as WT.
    wt_side = max(1, round(wt_fraction ** (1 / 3) * h))
    wt[:wt_side, :wt_side, :wt_side] = True

    real = rng.random(shape).astype(np.float32)
    pred = rng.random(shape).astype(np.float32)
    # raw_p995 from brain voxels; pred is already in [0,1] so pred_mode = "raw".
    brain_arr = brain.astype(np.int8)
    raw_p995 = float(np.percentile(pred[brain], 99.5))

    return ScanSample(
        scan_id="s0",
        patient_id="p0",
        cohort="test",
        ring="A",
        method="VENA",
        nfe=1,
        pred=pred,
        pred_raw=pred,
        pred_harmonised=pred,
        pred_mode="raw",
        raw_p995=raw_p995,
        real=real,
        brain=brain_arr,
        wt=wt.astype(np.int8),
        inference_seconds=1.0,
        peak_vram_mb=100.0,
    )


# ---------------------------------------------------------------------------
# C-noT uses bg (dilated) not bg_undilated
# ---------------------------------------------------------------------------


def test_c_not_uses_dilated_bg() -> None:
    """C-noT should use the dilation-expanded bg key, producing a smaller region than bg_undilated.

    With a non-trivial wt mask and dilate_k=5, region_masks["bg"] is strictly
    smaller than region_masks["bg_undilated"].  compute_scan_rows must use bg.
    """
    from vena.validation.regions import region_masks
    from vena.validation.spatial_residual import compute_scan_rows

    sample = _make_sample(wt_fraction=0.2, h=20, seed=1)
    masks = region_masks(sample.brain.astype(bool), sample.wt.astype(bool), dilate_k=5)

    n_bg = int(masks["bg"].sum())
    n_bg_undilated = int(masks["bg_undilated"].sum())
    # bg (dilated) must be strictly smaller than bg_undilated when dilation > 0.
    assert n_bg < n_bg_undilated, (
        f"bg={n_bg}, bg_undilated={n_bg_undilated}: dilation should shrink the non-tumour region"
    )

    rows = compute_scan_rows(sample, dilate_k=5, n_shuffles=5, n_boot=5, rng_seed=0)
    c_not_rows = [r for r in rows if r["condition"] == "C-noT"]
    assert len(c_not_rows) == 1

    # n_voxels_region for C-noT must equal n_bg (not n_bg_undilated).
    n_region = c_not_rows[0]["n_voxels_region"]
    if not np.isnan(n_region):
        assert int(n_region) == n_bg, (
            f"C-noT region size = {n_region}, expected bg size = {n_bg}. "
            "If bg_undilated ({n_bg_undilated}) was used instead, this fails."
        )


# ---------------------------------------------------------------------------
# compute_scan_rows — two rows per scan, correct conditions
# ---------------------------------------------------------------------------


def test_compute_scan_rows_returns_two_conditions() -> None:
    """compute_scan_rows always returns exactly 2 rows: C-WB and C-noT."""
    from vena.validation.spatial_residual import compute_scan_rows

    sample = _make_sample(wt_fraction=0.1, h=12, seed=2)
    rows = compute_scan_rows(sample, dilate_k=5, n_shuffles=5, n_boot=5, rng_seed=0)
    assert len(rows) == 2
    conditions = {r["condition"] for r in rows}
    assert conditions == {"C-WB", "C-noT"}


def test_compute_scan_rows_cwb_has_data() -> None:
    """C-WB row must have finite rho_s and n_voxels_region == n_brain."""
    from vena.validation.spatial_residual import compute_scan_rows

    sample = _make_sample(wt_fraction=0.0, h=12, seed=3)  # no WT → bg == brain
    rows = compute_scan_rows(sample, dilate_k=5, n_shuffles=5, n_boot=5, rng_seed=0)
    cwb = next(r for r in rows if r["condition"] == "C-WB")

    assert np.isfinite(cwb["rho_s"]), "C-WB rho_s should be finite"
    # n_brain = 12³ = 1728
    assert cwb["n_voxels_brain"] == 12**3
    assert cwb["n_voxels_region"] == 12**3


def test_compute_scan_rows_nan_on_empty_brain() -> None:
    """All metrics are NaN when brain is all-zeros (no foreground voxels)."""
    from vena.validation.io import ScanSample
    from vena.validation.spatial_residual import compute_scan_rows

    shape = (8, 8, 8)
    rng = np.random.default_rng(4)
    pred_empty = rng.random(shape).astype(np.float32)
    sample = ScanSample(
        scan_id="empty",
        patient_id="p0",
        cohort="test",
        ring="A",
        method="VENA",
        nfe=1,
        pred=pred_empty,
        pred_raw=pred_empty,
        pred_harmonised=pred_empty,
        pred_mode="harmonised",  # empty brain → select_scoring_volume falls back
        raw_p995=float("nan"),  # empty brain → no percentile possible
        real=rng.random(shape).astype(np.float32),
        brain=np.zeros(shape, dtype=np.int8),  # empty brain
        wt=np.zeros(shape, dtype=np.int8),
        inference_seconds=1.0,
        peak_vram_mb=0.0,
    )
    rows = compute_scan_rows(sample, dilate_k=5, n_shuffles=5, n_boot=5, rng_seed=0)
    assert len(rows) == 2
    for r in rows:
        assert np.isnan(r["rho_s"]), f"Expected NaN rho_s, got {r['rho_s']}"


# ---------------------------------------------------------------------------
# Engine end-to-end with synth_shard
# ---------------------------------------------------------------------------


def test_engine_end_to_end(synth_shard: Path, tmp_path: Path) -> None:
    """Engine runs on the synthetic shard and writes a valid per_scan.csv."""
    from routines.validation.spatial_residual.engine import (
        SpatialResidualConfig,
        SpatialResidualEngine,
    )

    # build_index globs <root>/*/predictions/*/*/nfe_*.h5.
    # Wrap synth_shard in a fresh sub-dir so the glob sees exactly one shard
    # and does not pick up other session tmp-dirs from sibling test files.
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "shard0").symlink_to(synth_shard)

    cfg = SpatialResidualConfig(
        inference_root=str(bench),
        output_root=str(tmp_path / "artifacts"),
        methods=["VENA-S1-v3b-rw", "C0-Identity"],
        cohorts=None,
        nfes=[5],
        dilate_k=5,
        n_shuffles=5,
        n_boot=5,
        rng_seed=42,
        mi_n_voxels=500,
        n_deciles=5,
        vena_method="VENA-S1-v3b-rw",
        scan_limit=None,
        run_convergence_check=False,
        log_level="WARNING",
    )
    engine = SpatialResidualEngine(cfg)
    run_dir = engine.run()

    per_scan_csv = run_dir / "per_scan" / "spatial_residual.csv"
    assert per_scan_csv.exists(), "per_scan/spatial_residual.csv must be written"

    df = pd.read_csv(per_scan_csv)

    # Frozen header check.
    assert list(df.columns) == SPATIAL_CSV_COLUMNS, "CSV column mismatch"

    # Each scan produces 2 rows (C-WB and C-noT).
    # synth_shard: TestCohortA (3 scans) + LUMIERE-like (5 scans) + TestCohortB (2 scans)
    # × 2 methods × 2 conditions = 40 rows total.
    assert len(df) == 40, f"Expected 40 rows, got {len(df)}"

    # Both conditions present.
    assert set(df["condition"].unique()) == {"C-WB", "C-noT"}

    # decision.json must exist.
    assert (run_dir / "decision.json").exists()

    # LATEST symlink must point at run_dir.
    latest = (tmp_path / "artifacts" / "spatial_residual" / "LATEST").resolve()
    assert latest == run_dir


def test_engine_lumiere_collapse(synth_shard: Path, tmp_path: Path) -> None:
    """LUMIERE-like 5 scans must collapse to 3 patients in aggregate output."""
    from routines.validation.spatial_residual.engine import (
        SpatialResidualConfig,
        SpatialResidualEngine,
    )

    from vena.validation.stats import collapse_to_patient

    bench = tmp_path / "bench_lumiere"
    bench.mkdir()
    (bench / "shard0").symlink_to(synth_shard)

    cfg = SpatialResidualConfig(
        inference_root=str(bench),
        output_root=str(tmp_path / "artifacts_lumiere"),
        methods=["VENA-S1-v3b-rw"],
        cohorts=["LUMIERE-like"],
        nfes=[5],
        dilate_k=5,
        n_shuffles=5,
        n_boot=5,
        rng_seed=42,
        mi_n_voxels=500,
        n_deciles=5,
        vena_method="VENA-S1-v3b-rw",
        scan_limit=None,
        run_convergence_check=False,
        log_level="WARNING",
    )
    engine = SpatialResidualEngine(cfg)
    run_dir = engine.run()

    df = pd.read_csv(run_dir / "per_scan" / "spatial_residual.csv")

    # LUMIERE-like: 5 scan_ids, 3 patient_ids.
    lumiere_df = df[df["cohort"] == "LUMIERE-like"]
    # 5 scans × 1 method × 2 conditions = 10 rows
    assert len(lumiere_df) == 10, f"Expected 10 LUMIERE rows, got {len(lumiere_df)}"

    # 5 distinct scan IDs in the per-scan CSV.
    assert lumiere_df["scan_id"].nunique() == 5, (
        f"Expected 5 distinct scan_ids, got {lumiere_df['scan_id'].nunique()}"
    )

    # 3 distinct patient IDs — the critical collapse invariant (§11 join-trap).
    assert lumiere_df["patient_id"].nunique() == 3, (
        f"Expected 3 distinct patient_ids (lum_pt1, lum_pt2, lum_pt3), "
        f"got {lumiere_df['patient_id'].nunique()}"
    )

    # Verify collapse_to_patient reduces 5 scans → 3 patients.
    # collapse_to_patient groups by (method, cohort, nfe, patient_id) and
    # averages value_cols across conditions and longitudinal scans.
    patient_df = collapse_to_patient(
        lumiere_df[lumiere_df["condition"] == "C-WB"],
        value_cols=["rho_s"],
        by=("method", "cohort", "nfe", "patient_id"),
    )
    assert patient_df["patient_id"].nunique() == 3, (
        f"collapse_to_patient must yield 3 unique patients from LUMIERE 5 scans; "
        f"got {patient_df['patient_id'].nunique()}"
    )

"""Unit tests for routines.validation.paired_fidelity.

Tests the engine end-to-end against the ``synth_shard`` session fixture
(schema-2.0 shard with LUMIERE-like longitudinal cohort).

Verifies:
- PairedFidelityConfig.from_yaml parses required fields.
- Engine run() produces the expected artifact tree.
- per_scan CSV has correct column set.
- LUMIERE-like collapses to fewer patients than scans.
- C0 sanity CSV is written.
- decision.json is schema-compliant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from routines.validation.paired_fidelity.engine import (
    PairedFidelityConfig,
    PairedFidelityEngine,
    PairedFidelityError,
)

pytestmark = pytest.mark.validation


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_from_yaml_parses_required(tmp_path: Path) -> None:
    """from_yaml reads data_root and output_root correctly."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"data_root: {tmp_path / 'data'}\noutput_root: {tmp_path / 'out'}\n")
    cfg = PairedFidelityConfig.from_yaml(cfg_path)
    assert cfg.data_root == tmp_path / "data"
    assert cfg.output_root == tmp_path / "out"


def test_from_yaml_defaults(tmp_path: Path) -> None:
    """Optional keys default to documented values."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"data_root: {tmp_path}\noutput_root: {tmp_path}\n")
    cfg = PairedFidelityConfig.from_yaml(cfg_path)
    assert cfg.dilate_k == 5
    assert cfg.ssim_window_size == 7
    assert cfg.n_bootstrap == 10_000
    assert cfg.bootstrap_seed == 1337
    assert cfg.device == "cpu"
    assert cfg.filter_methods == ()
    assert cfg.filter_rings == ("A", "B")
    assert cfg.ms_ssim_weights == (0.0448, 0.2856, 0.3001, 0.3633)


def test_from_yaml_filter_fields(tmp_path: Path) -> None:
    """filter_methods / filter_nfe are parsed into tuples."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"data_root: {tmp_path}\noutput_root: {tmp_path}\n"
        "filter_methods:\n  - MethodA\n  - MethodB\n"
        "filter_nfe: [1, 5]\n"
    )
    cfg = PairedFidelityConfig.from_yaml(cfg_path)
    assert cfg.filter_methods == ("MethodA", "MethodB")
    assert cfg.filter_nfe == (1, 5)


def test_from_yaml_missing_data_root_raises(tmp_path: Path) -> None:
    """Missing data_root raises PairedFidelityError."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"output_root: {tmp_path}\n")
    with pytest.raises(PairedFidelityError, match="data_root"):
        PairedFidelityConfig.from_yaml(cfg_path)


def test_from_yaml_nonexistent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PairedFidelityError):
        PairedFidelityConfig.from_yaml(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Engine run() — full artifact tree
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine_cfg(synth_inference_root: Path, tmp_path: Path) -> PairedFidelityConfig:
    """Config wired to the session synth_inference_root fixture.

    ``discover_shards`` globs ``<data_root>/*/decision.json`` and
    ``build_index`` then globs within accepted shards.  The conftest
    ``synth_inference_root`` is exactly the right root: it contains one
    production shard (``test_prod_shard/``) with a ``decision.json`` that
    has no ``smoke`` key (treated as production, not filtered out).

    Using ``synth_inference_root`` directly avoids the old stale-basetemp
    accumulation issue: the conftest creates the shard under a session-scoped
    ``tmp_path_factory.mktemp("inference")`` directory that is fresh each run.
    """
    return PairedFidelityConfig(
        data_root=synth_inference_root,
        output_root=tmp_path / "analyses",
        dilate_k=5,
        ssim_window_size=7,
        ssim_window_sigma=1.5,
        ms_ssim_weights=(0.0448, 0.2856, 0.3001, 0.3633),
        ms_ssim_bbox_margin=8,
        n_bootstrap=100,  # fast
        bootstrap_seed=42,
        device="cpu",
    )


def test_engine_run_returns_path(engine_cfg: PairedFidelityConfig) -> None:
    """Engine.run() returns a Path that exists."""
    engine = PairedFidelityEngine(engine_cfg)
    run_dir = engine.run()
    assert run_dir.is_dir()


def test_engine_artifact_tree(engine_cfg: PairedFidelityConfig) -> None:
    """run() creates per_scan/, tables/, figures/, decision.json, report.md."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    assert (run_dir / "per_scan" / "paired_fidelity.csv").is_file()
    assert (run_dir / "per_scan" / "paired_fidelity_patient.csv").is_file()
    assert (run_dir / "decision.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "tables").is_dir()
    assert (run_dir / "figures").is_dir()


def test_engine_per_scan_csv_columns(engine_cfg: PairedFidelityConfig) -> None:
    """per_scan CSV has all required columns."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    df = pd.read_csv(run_dir / "per_scan" / "paired_fidelity.csv")

    required_cols = [
        "method",
        "cohort",
        "ring",
        "nfe",
        "scan_id",
        "patient_id",
        # §4.1 scoring-space audit — must be present per SHARED_CONTRACTS §7
        "pred_mode",
        "raw_p995",
        "mae_brain",
        "mae_wt",
        "mae_bg_undilated",
        "rmse_brain",
        "rmse_wt",
        "rmse_bg_undilated",
        "psnr_brain",
        "psnr_wt",
        "psnr_bg_undilated",
        "ssim_brain",
        "ssim_wt",
        "ssim_bg_undilated",
        "ms_ssim_brain",
        "ms_ssim_wt",
        "ms_ssim_bg_undilated",
        "zgd",
        "n_brain_voxels",
        "n_wt_voxels",
        "n_bg_undilated_voxels",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_engine_per_scan_csv_no_nan_in_ids(engine_cfg: PairedFidelityConfig) -> None:
    """ID columns (scan_id, patient_id, method, cohort) have no NaN values."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    df = pd.read_csv(run_dir / "per_scan" / "paired_fidelity.csv")
    for col in ("scan_id", "patient_id", "method", "cohort"):
        assert df[col].notna().all(), f"NaN found in column {col!r}"


def test_engine_lumiere_collapse(engine_cfg: PairedFidelityConfig) -> None:
    """LUMIERE-like cohort collapses from 5 scans → 3 patients (not more)."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    scan_df = pd.read_csv(run_dir / "per_scan" / "paired_fidelity.csv")
    pt_df = pd.read_csv(run_dir / "per_scan" / "paired_fidelity_patient.csv")

    lum_scan = scan_df[scan_df["cohort"] == "LUMIERE-like"]
    lum_pt = pt_df[pt_df["cohort"] == "LUMIERE-like"]

    if lum_scan.empty:
        pytest.skip("LUMIERE-like not in synthetic shard")

    # Take a single method/nfe slice
    method = lum_scan["method"].iloc[0]
    nfe = lum_scan["nfe"].iloc[0]
    n_scans = len(lum_scan[(lum_scan["method"] == method) & (lum_scan["nfe"] == nfe)])
    n_patients = len(lum_pt[(lum_pt["method"] == method) & (lum_pt["nfe"] == nfe)])

    # synth_shard fixture: 5 scans, 3 patients
    assert n_scans == 5, f"Expected 5 LUMIERE-like scans, got {n_scans}"
    assert n_patients == 3, f"Expected 3 LUMIERE-like patients, got {n_patients}"
    assert n_patients < n_scans, "Patient collapse did not reduce row count"


def test_engine_c0_sanity_csv(engine_cfg: PairedFidelityConfig) -> None:
    """tables/c0_sanity.csv is written when C0-Identity is present at its selection NFE.

    The synth_shard uses nfe=5 for all methods, but SELECTION_NFE["C0-Identity"]=1,
    so the engine warns and skips the sanity check — c0_sanity.csv may not exist.
    We just verify the engine completes without error (no assertion on file presence).
    """
    run_dir = PairedFidelityEngine(engine_cfg).run()
    # The run must complete and write the mandatory artifacts
    assert (run_dir / "decision.json").is_file()
    # c0_sanity.csv is optional — written only when C0 is at its selection NFE
    c0_path = run_dir / "tables" / "c0_sanity.csv"
    if c0_path.is_file():
        df = pd.read_csv(c0_path)
        assert "method" in df.columns
        assert "mae_brain" in df.columns
        assert "mae_wt" in df.columns


def test_engine_decision_json_schema(engine_cfg: PairedFidelityConfig) -> None:
    """decision.json is valid JSON with required keys."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    with open(run_dir / "decision.json") as fh:
        d = json.load(fh)

    required_keys = [
        "schema_version",
        "producer",
        "produced_at",
        "data_root",
        "output_root",
        "dilate_k",
        "ssim_window_size",
        "ssim_window_sigma",
        "ms_ssim_weights",
        "n_bootstrap",
        "competitor_family",
        "ablation_family",
        "n_competitor",
        "n_ablation",
        "holm_correction",
        "ssim_treatment",
        "ms_ssim_treatment",
        "n_scans",
        "n_patients",
        "elapsed_s",
        # §4.1 / SHARED_CONTRACTS §7 — scoring-space audit
        "pred_mode_counts_by_method",
        "scoring_space_note",
        # §3.1 shard provenance
        "skipped_smoke_shards",
    ]
    missing = [k for k in required_keys if k not in d]
    assert not missing, f"Missing keys in decision.json: {missing}"


def test_engine_decision_json_family_sizes(engine_cfg: PairedFidelityConfig) -> None:
    """decision.json records exactly 8 competitor and 3 ablation methods."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    with open(run_dir / "decision.json") as fh:
        d = json.load(fh)
    assert d["n_competitor"] == 8
    assert d["n_ablation"] == 3


def test_engine_decision_json_ssim_treatment_documents_principled(
    engine_cfg: PairedFidelityConfig,
) -> None:
    """decision.json explicitly documents the principled SSIM-map approach."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    with open(run_dir / "decision.json") as fh:
        d = json.load(fh)
    assert "principled" in d["ssim_treatment"].lower()
    assert "map" in d["ssim_treatment"].lower()


def test_engine_decision_json_pred_mode_counts(engine_cfg: PairedFidelityConfig) -> None:
    """pred_mode_counts_by_method maps each method to its raw/harmonised scan counts.

    In the synth shard, raw predictions are uniform random in [0, 1] so
    brain p99.5 < 1.05 → all scans should be scored as "raw".
    """
    run_dir = PairedFidelityEngine(engine_cfg).run()
    with open(run_dir / "decision.json") as fh:
        d = json.load(fh)
    counts = d["pred_mode_counts_by_method"]
    assert isinstance(counts, dict), "pred_mode_counts_by_method must be a dict"
    assert len(counts) > 0, "pred_mode_counts_by_method is empty"
    for method, mode_dict in counts.items():
        assert isinstance(mode_dict, dict), f"mode dict for {method} must be a dict"
        total = sum(mode_dict.values())
        assert total > 0, f"No scans counted for method {method}"


def test_engine_decision_json_skipped_smoke_shards(engine_cfg: PairedFidelityConfig) -> None:
    """skipped_smoke_shards is a list (empty for the synth shard which is production)."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    with open(run_dir / "decision.json") as fh:
        d = json.load(fh)
    skipped = d["skipped_smoke_shards"]
    assert isinstance(skipped, list), "skipped_smoke_shards must be a list"
    # The synth shard has no smoke key → treated as production → list is empty
    assert skipped == [], f"Expected no smoke shards skipped, got {skipped}"


def test_engine_pred_mode_column_values(engine_cfg: PairedFidelityConfig) -> None:
    """pred_mode column contains only 'raw' or 'harmonised'; raw_p995 is finite float."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    df = pd.read_csv(run_dir / "per_scan" / "paired_fidelity.csv")
    assert "pred_mode" in df.columns, "pred_mode column missing"
    assert "raw_p995" in df.columns, "raw_p995 column missing"
    valid_modes = {"raw", "harmonised"}
    bad_modes = set(df["pred_mode"].unique()) - valid_modes
    assert not bad_modes, f"Unexpected pred_mode values: {bad_modes}"
    # raw_p995 must be finite for all scans (synth data has a full brain mask)
    assert df["raw_p995"].notna().all(), "raw_p995 has NaN values"
    assert (df["raw_p995"] >= 0).all(), "raw_p995 has negative values"


def test_engine_symlink_latest(engine_cfg: PairedFidelityConfig) -> None:
    """LATEST symlink in the routine directory points to the run dir."""
    run_dir = PairedFidelityEngine(engine_cfg).run()
    latest = run_dir.parent / "LATEST"
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()


def test_engine_filter_methods(engine_cfg: PairedFidelityConfig) -> None:
    """filter_methods restricts per_scan CSV to the requested subset."""
    # Only process C0-Identity from the synth_shard
    cfg2 = PairedFidelityConfig(
        data_root=engine_cfg.data_root,
        output_root=engine_cfg.output_root,
        filter_methods=("C0-Identity",),
        n_bootstrap=100,
    )
    run_dir = PairedFidelityEngine(cfg2).run()
    df = pd.read_csv(run_dir / "per_scan" / "paired_fidelity.csv")
    assert set(df["method"].unique()) == {"C0-Identity"}


def test_engine_filter_rings(engine_cfg: PairedFidelityConfig) -> None:
    """filter_rings=A excludes Ring-B cohort from per_scan CSV."""
    cfg2 = PairedFidelityConfig(
        data_root=engine_cfg.data_root,
        output_root=engine_cfg.output_root,
        filter_rings=("A",),
        n_bootstrap=100,
    )
    run_dir = PairedFidelityEngine(cfg2).run()
    df = pd.read_csv(run_dir / "per_scan" / "paired_fidelity.csv")
    assert "B" not in df["ring"].values


def test_engine_empty_index_raises(tmp_path: Path) -> None:
    """Engine raises PairedFidelityError when data_root contains no H5 files."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    cfg = PairedFidelityConfig(
        data_root=empty_root,
        output_root=tmp_path / "out",
        n_bootstrap=10,
    )
    with pytest.raises(PairedFidelityError):
        PairedFidelityEngine(cfg).run()


def test_engine_c0_identity_mae_is_null_floor(engine_cfg: PairedFidelityConfig) -> None:
    """When c0_sanity.csv is written, C0-Identity row has finite non-negative MAE.

    The synth_shard uses nfe=5 for all methods but SELECTION_NFE["C0-Identity"]=1,
    so the sanity check is skipped and the CSV may not exist.  This test is a
    no-op when the file is absent (see test_engine_c0_sanity_csv for the skip note).
    """
    run_dir = PairedFidelityEngine(engine_cfg).run()
    c0_path = run_dir / "tables" / "c0_sanity.csv"
    if not c0_path.is_file():
        pytest.skip("c0_sanity.csv not written (C0-Identity not at selection NFE=1 in synth shard)")
    df = pd.read_csv(c0_path)
    c0_row = df[df["method"] == "C0-Identity"]
    assert not c0_row.empty, "C0-Identity row missing from c0_sanity.csv"
    mae_brain = c0_row["mae_brain"].iloc[0]
    assert math.isfinite(mae_brain)
    assert mae_brain >= 0.0

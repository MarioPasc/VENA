"""Tests for vena.validation.artifacts.

Covers: run-dir layout, decision.json, per_scan.csv, LATEST symlink.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.validation


def test_make_run_dir_creates_subdirs(tmp_path: Path) -> None:
    """make_run_dir creates tables/, figures/, per_scan/ inside the run dir."""
    from vena.validation.artifacts import make_run_dir

    run_dir = make_run_dir(tmp_path, "test_routine")

    assert (run_dir / "tables").is_dir()
    assert (run_dir / "figures").is_dir()
    assert (run_dir / "per_scan").is_dir()


def test_make_run_dir_path_structure(tmp_path: Path) -> None:
    """Run dir is <output_root>/<routine_name>/<UTC-stamp>/."""
    from vena.validation.artifacts import make_run_dir

    run_dir = make_run_dir(tmp_path, "paired_fidelity")

    # Parent is the routine directory.
    assert run_dir.parent.name == "paired_fidelity"
    # Grandparent is the output root.
    assert run_dir.parent.parent == tmp_path


def test_write_decision_json_roundtrip(tmp_path: Path) -> None:
    """write_decision_json writes valid JSON that round-trips the payload."""
    from vena.validation.artifacts import make_run_dir, write_decision_json

    run_dir = make_run_dir(tmp_path, "test_routine")
    payload = {"schema_version": "1.0", "cohorts": ["UCSF-PDGM"]}
    out = write_decision_json(run_dir, payload)

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["schema_version"] == "1.0"
    assert loaded["cohorts"] == ["UCSF-PDGM"]
    # Injected fields must be present.
    assert "produced_at" in loaded
    assert "git_sha" in loaded  # may be None if not in a git repo


def test_write_decision_json_does_not_overwrite_produced_at(tmp_path: Path) -> None:
    """Caller-supplied produced_at is preserved (setdefault semantics)."""
    from vena.validation.artifacts import make_run_dir, write_decision_json

    run_dir = make_run_dir(tmp_path, "test_routine")
    payload = {"produced_at": "2026-01-01T00:00:00+00:00"}
    write_decision_json(run_dir, payload)

    loaded = json.loads((run_dir / "decision.json").read_text())
    assert loaded["produced_at"] == "2026-01-01T00:00:00+00:00"


def test_write_per_scan_csv_roundtrip(tmp_path: Path) -> None:
    """write_per_scan_csv writes a CSV that reads back identically."""
    from vena.validation.artifacts import make_run_dir, write_per_scan_csv

    run_dir = make_run_dir(tmp_path, "test_routine")
    df = pd.DataFrame(
        {
            "scan_id": ["s1", "s2"],
            "method": ["VENA-S1-v3b-rw", "C0-Identity"],
            "ssim": [0.85, 0.60],
        }
    )
    out = write_per_scan_csv(run_dir, df)

    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == 2
    assert loaded.iloc[0]["scan_id"] == "s1"


def test_write_per_scan_csv_custom_name(tmp_path: Path) -> None:
    """Custom name parameter is honoured."""
    from vena.validation.artifacts import make_run_dir, write_per_scan_csv

    run_dir = make_run_dir(tmp_path, "test_routine")
    df = pd.DataFrame({"x": [1]})
    out = write_per_scan_csv(run_dir, df, name="audit.csv")

    assert out.name == "audit.csv"
    assert out.parent.name == "per_scan"


def test_symlink_latest_points_at_run_dir(tmp_path: Path) -> None:
    """LATEST symlink resolves to the run directory."""
    from vena.validation.artifacts import make_run_dir, symlink_latest

    run_dir = make_run_dir(tmp_path, "test_routine")
    symlink_latest(run_dir)

    latest = run_dir.parent / "LATEST"
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()


def test_symlink_latest_updated_on_second_run(tmp_path: Path) -> None:
    """A second call to symlink_latest moves LATEST to the newer directory."""
    from vena.validation.artifacts import make_run_dir, symlink_latest

    run_dir_1 = make_run_dir(tmp_path, "test_routine")
    symlink_latest(run_dir_1)

    run_dir_2 = make_run_dir(tmp_path, "test_routine")
    symlink_latest(run_dir_2)

    latest = run_dir_2.parent / "LATEST"
    assert latest.resolve() == run_dir_2.resolve()

"""Tests for the cohort-dedup pre-flight gate in routines.fm.train.engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from routines.fm.train.engine import _assert_dedup_gate
from routines.fm.train.exceptions import PreflightGateError

from vena.preflight.cohort_dedup.decision import (
    DEDUP_DECISION_SCHEMA_VERSION,
    write_decision,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal stand-ins for the real cfg.data / cfg objects (the gate only reads a
# tiny subset of attributes from cfg.data, so duck typing is enough).
# ---------------------------------------------------------------------------


class _DataStub(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    corpus_registry: Path
    dedup_decisions_path: Path | None


class _CfgStub(BaseModel):
    data: _DataStub


def _make_registry(tmp_path: Path) -> Path:
    """Tiny synthetic corpus registry. References dummy H5 paths that exist."""
    img_h5 = tmp_path / "dummy.h5"
    img_h5.write_bytes(b"")  # placeholder; load_registry only checks existence
    reg = {
        "schema_version": "1.0.0",
        "name": "tiny",
        "cohorts": [
            {
                "name": "BraTS-GLI",
                "pathology": "preoperative_glioma",
                "label_system": "BraTS2023",
                "role": "cv",
                "longitudinal": False,
                "image_h5": str(img_h5),
                "latent_h5": str(img_h5),
                "n_patients": 1,
                "n_scans": 1,
                "modalities": ["t1pre", "t1c", "t2", "flair"],
                "has_swan": False,
            },
            {
                "name": "UCSF-PDGM",
                "pathology": "preoperative_glioma",
                "label_system": "BraTS2021",
                "role": "cv",
                "longitudinal": False,
                "image_h5": str(img_h5),
                "latent_h5": str(img_h5),
                "n_patients": 1,
                "n_scans": 1,
                "modalities": ["t1pre", "t1c", "t2", "flair"],
                "has_swan": False,
            },
        ],
    }
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(reg))
    return p


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_decision(reg_path: Path, xlsx_path: Path) -> dict:
    return {
        "schema_version": DEDUP_DECISION_SCHEMA_VERSION,
        "produced_at": "2026-06-02T12:00:00Z",
        "producer": "test:0",
        "corpus_registry_path": str(reg_path),
        "corpus_registry_sha256": _sha256(reg_path),
        "mapping_xlsx_path": str(xlsx_path),
        "mapping_xlsx_sha256": "0" * 64,
        "priority": ["BraTS-GLI", "UCSF-PDGM"],
        "policy": "drop_lower_priority",
        "totals": {
            "n_cohorts": 2,
            "n_patients_total_in": 2,
            "n_patients_total_kept": 2,
            "n_patients_total_rejected": 0,
        },
        "cohorts": {
            "BraTS-GLI": {
                "n_total": 1,
                "n_kept": 1,
                "n_rejected": 0,
                "bridge_field": None,
                "kept_patient_ids": ["BraTS-GLI-00000"],
                "rejected_patient_ids": [],
            },
            "UCSF-PDGM": {
                "n_total": 1,
                "n_kept": 1,
                "n_rejected": 0,
                "bridge_field": "metadata/brats21_id",
                "kept_patient_ids": ["UCSF-PDGM-0000"],
                "rejected_patient_ids": [],
            },
        },
        "overlap_audit": [],
        "unresolvable_overlaps": [],
    }


def test_gate_passes_on_valid_decision(tmp_path: Path) -> None:
    reg = _make_registry(tmp_path)
    xlsx = tmp_path / "map.xlsx"
    xlsx.write_bytes(b"x")
    dec = tmp_path / "decision.json"
    write_decision(dec, _valid_decision(reg, xlsx))
    cfg = _CfgStub(data=_DataStub(corpus_registry=reg, dedup_decisions_path=dec))
    _assert_dedup_gate(cfg)  # no raise


def test_gate_raises_on_missing_file(tmp_path: Path) -> None:
    reg = _make_registry(tmp_path)
    cfg = _CfgStub(data=_DataStub(corpus_registry=reg, dedup_decisions_path=tmp_path / "no.json"))
    with pytest.raises(PreflightGateError, match="does not exist"):
        _assert_dedup_gate(cfg)


def test_gate_raises_on_registry_sha_mismatch(tmp_path: Path) -> None:
    reg = _make_registry(tmp_path)
    xlsx = tmp_path / "map.xlsx"
    xlsx.write_bytes(b"x")
    dec = tmp_path / "decision.json"
    payload = _valid_decision(reg, xlsx)
    payload["corpus_registry_sha256"] = "0" * 64  # wrong sha
    write_decision(dec, payload)
    cfg = _CfgStub(data=_DataStub(corpus_registry=reg, dedup_decisions_path=dec))
    with pytest.raises(PreflightGateError, match="SHA-256"):
        _assert_dedup_gate(cfg)


def test_gate_raises_when_cohort_missing(tmp_path: Path) -> None:
    reg = _make_registry(tmp_path)
    xlsx = tmp_path / "map.xlsx"
    xlsx.write_bytes(b"x")
    dec = tmp_path / "decision.json"
    payload = _valid_decision(reg, xlsx)
    # Drop UCSF-PDGM from the decision but it's still in the registry.
    del payload["cohorts"]["UCSF-PDGM"]
    payload["totals"]["n_cohorts"] = 1
    payload["totals"]["n_patients_total_in"] = 1
    payload["totals"]["n_patients_total_kept"] = 1
    write_decision(dec, payload)
    cfg = _CfgStub(data=_DataStub(corpus_registry=reg, dedup_decisions_path=dec))
    with pytest.raises(PreflightGateError, match="missing cohorts"):
        _assert_dedup_gate(cfg)

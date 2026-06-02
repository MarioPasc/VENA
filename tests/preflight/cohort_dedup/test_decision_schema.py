"""Round-trip + validation tests for the decision.json schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vena.preflight.cohort_dedup.decision import (
    DEDUP_DECISION_SCHEMA_VERSION,
    DedupDecisionSchemaError,
    assert_dedup_decision_valid,
    build_allowlists,
    write_decision,
)

pytestmark = pytest.mark.unit


def _minimal_payload() -> dict:
    return {
        "schema_version": DEDUP_DECISION_SCHEMA_VERSION,
        "produced_at": "2026-06-02T12:00:00Z",
        "producer": "test:0",
        "corpus_registry_path": "/tmp/corpus.json",
        "corpus_registry_sha256": "deadbeef" * 8,
        "mapping_xlsx_path": "/tmp/map.xlsx",
        "mapping_xlsx_sha256": "cafebabe" * 8,
        "priority": ["BraTS-GLI", "UCSF-PDGM"],
        "policy": "drop_lower_priority",
        "totals": {
            "n_cohorts": 2,
            "n_patients_total_in": 5,
            "n_patients_total_kept": 4,
            "n_patients_total_rejected": 1,
        },
        "cohorts": {
            "BraTS-GLI": {
                "n_total": 2,
                "n_kept": 2,
                "n_rejected": 0,
                "bridge_field": None,
                "kept_patient_ids": ["BraTS-GLI-00000", "BraTS-GLI-00001"],
                "rejected_patient_ids": [],
            },
            "UCSF-PDGM": {
                "n_total": 3,
                "n_kept": 2,
                "n_rejected": 1,
                "bridge_field": "metadata/brats21_id",
                "kept_patient_ids": ["UCSF-PDGM-0001", "UCSF-PDGM-0002"],
                "rejected_patient_ids": ["UCSF-PDGM-0000"],
            },
        },
        "overlap_audit": [],
        "unresolvable_overlaps": [],
    }


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "decision.json"
    write_decision(p, _minimal_payload())
    parsed = assert_dedup_decision_valid(p)
    assert parsed["cohorts"]["UCSF-PDGM"]["n_rejected"] == 1


def test_build_allowlists() -> None:
    allow = build_allowlists(_minimal_payload())
    assert allow["UCSF-PDGM"] == {"UCSF-PDGM-0001", "UCSF-PDGM-0002"}
    assert allow["BraTS-GLI"] == {"BraTS-GLI-00000", "BraTS-GLI-00001"}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DedupDecisionSchemaError, match="not found"):
        assert_dedup_decision_valid(tmp_path / "missing.json")


def test_wrong_schema_version_raises(tmp_path: Path) -> None:
    p = tmp_path / "decision.json"
    bad = _minimal_payload() | {"schema_version": "9.9"}
    write_decision(p, bad)
    with pytest.raises(DedupDecisionSchemaError, match="schema_version"):
        assert_dedup_decision_valid(p)


def test_cohort_total_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "decision.json"
    bad = _minimal_payload()
    bad["cohorts"]["UCSF-PDGM"]["n_total"] = 7  # 7 != 2 + 1
    write_decision(p, bad)
    with pytest.raises(DedupDecisionSchemaError, match="n_total"):
        assert_dedup_decision_valid(p)


def test_kept_length_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "decision.json"
    bad = _minimal_payload()
    bad["cohorts"]["UCSF-PDGM"]["kept_patient_ids"] = ["UCSF-PDGM-0001"]  # len=1 != n_kept=2
    write_decision(p, bad)
    with pytest.raises(DedupDecisionSchemaError, match="kept_patient_ids"):
        assert_dedup_decision_valid(p)


def test_missing_top_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "decision.json"
    bad = _minimal_payload()
    del bad["priority"]
    p.write_text(json.dumps(bad))
    with pytest.raises(DedupDecisionSchemaError, match="priority"):
        assert_dedup_decision_valid(p)

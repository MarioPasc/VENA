"""The decision contract must round-trip through JSON and carry every key
consumed by the Phase-3 training engine."""

from __future__ import annotations

import json

from vena.preflight.priors_validation.core.dataclasses import (
    CohortReport,
    TestOutcome,
    ValidationResult,
)
from vena.preflight.priors_validation.reporting.cohort_summary import _decision_json


def _make_report() -> CohortReport:
    vr = ValidationResult(
        subject_id="SYNTH-0001",
        outcomes=(
            TestOutcome(
                test_id="T1_range_sanity",
                subject_id="SYNTH-0001",
                prior_id="cbf",
                roi_id="nawm",
                metric_name="median_cbf_nawm",
                metric_value=22.0,
                threshold=(12.0, 35.0),
                passed=True,
                severity="info",
                diagnostic="ok",
            ),
        ),
        overall_passed=True,
        failed_priors=frozenset(),
    )
    return CohortReport(
        n_subjects=1,
        n_subjects_applicable=1,
        per_test_pass_rate={
            "T1_range_sanity": 1.0,
            "T2_atlas_localisation": 1.0,
            "T3_t1gd_coherence": 1.0,
            "T4_cross_modal": 1.0,
            "T5_reproducibility": None,
        },
        per_prior_clearance={"cbf": "passed"},
        cohort_pass_rate_overall=1.0,
        training_clearance=True,
        subjects=(vr,),
        atlas_versions={"mni152_nlin2009c": "Fonov2011"},
        routine_version="0.1.0",
        warnings=(),
    )


def test_decision_json_round_trip():
    report = _make_report()
    payload = _decision_json(report)
    blob = json.dumps(payload)
    loaded = json.loads(blob)
    required = {
        "schema_version",
        "n_subjects",
        "cohort_pass_rate_overall",
        "per_test_pass_rate",
        "per_prior_clearance",
        "atlas_versions",
        "routine_version",
        "training_clearance",
    }
    assert required.issubset(loaded.keys())
    assert loaded["training_clearance"] is True
    assert loaded["per_prior_clearance"]["cbf"] == "passed"

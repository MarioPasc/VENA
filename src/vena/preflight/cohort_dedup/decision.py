"""decision.json schema + (de)serialiser for the cohort_dedup preflight.

Schema v1.0 contract consumed by
``routines.fm.train.engine._assert_preflight_gates``:

* ``schema_version`` (str) — must equal :data:`DEDUP_DECISION_SCHEMA_VERSION`.
* ``produced_at`` (ISO-8601 UTC).
* ``producer`` (str, ``"routines.preflights.cohort_dedup:<version>"``).
* ``corpus_registry_path`` + ``corpus_registry_sha256``.
* ``mapping_xlsx_path`` + ``mapping_xlsx_sha256``.
* ``priority`` (list[str]).
* ``policy`` (str, currently always ``"drop_lower_priority"``).
* ``totals`` (``{n_cohorts, n_patients_total_in/kept/rejected}``).
* ``cohorts`` (``{cohort -> {n_total, n_kept, n_rejected, bridge_field,
  kept_patient_ids, rejected_patient_ids, ...}}``).
* ``overlap_audit`` (list of resolved duplicate groups).
* ``unresolvable_overlaps`` (list).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEDUP_DECISION_SCHEMA_VERSION = "1.0"


class DedupDecisionSchemaError(Exception):
    """Raised when a decision.json fails schema validation."""


REQUIRED_TOP_KEYS: tuple[str, ...] = (
    "schema_version",
    "produced_at",
    "producer",
    "corpus_registry_path",
    "corpus_registry_sha256",
    "mapping_xlsx_path",
    "mapping_xlsx_sha256",
    "priority",
    "policy",
    "totals",
    "cohorts",
    "overlap_audit",
    "unresolvable_overlaps",
)

REQUIRED_COHORT_KEYS: tuple[str, ...] = (
    "n_total",
    "n_kept",
    "n_rejected",
    "bridge_field",
    "kept_patient_ids",
    "rejected_patient_ids",
)


def assert_dedup_decision_valid(path: Path | str) -> dict[str, Any]:
    """Load a decision.json and assert it conforms to schema v1.0.

    Parameters
    ----------
    path
        Path to ``decision.json``.

    Returns
    -------
    dict
        The parsed payload (on success).

    Raises
    ------
    DedupDecisionSchemaError
        If the file is missing, malformed, or violates the schema.
    """
    path = Path(path)
    if not path.is_file():
        raise DedupDecisionSchemaError(f"decision.json not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DedupDecisionSchemaError(f"decision.json is not valid JSON: {path}: {exc}") from exc

    violations: list[str] = []
    if payload.get("schema_version") != DEDUP_DECISION_SCHEMA_VERSION:
        violations.append(
            f"schema_version != {DEDUP_DECISION_SCHEMA_VERSION!r}; "
            f"got {payload.get('schema_version')!r}"
        )
    for k in REQUIRED_TOP_KEYS:
        if k not in payload:
            violations.append(f"missing top-level key: {k!r}")
    cohorts = payload.get("cohorts", {})
    if not isinstance(cohorts, dict) or not cohorts:
        violations.append("cohorts must be a non-empty object")
    else:
        for name, entry in cohorts.items():
            if not isinstance(entry, dict):
                violations.append(f"cohorts[{name!r}] must be an object")
                continue
            for k in REQUIRED_COHORT_KEYS:
                if k not in entry:
                    violations.append(f"cohorts[{name!r}] missing key: {k!r}")
            if "n_total" in entry and "n_kept" in entry and "n_rejected" in entry:
                if entry["n_total"] != entry["n_kept"] + entry["n_rejected"]:
                    violations.append(f"cohorts[{name!r}]: n_total != n_kept + n_rejected")
            kept = entry.get("kept_patient_ids")
            if not isinstance(kept, list):
                violations.append(f"cohorts[{name!r}].kept_patient_ids must be a list")
            elif "n_kept" in entry and len(kept) != entry["n_kept"]:
                violations.append(
                    f"cohorts[{name!r}]: len(kept_patient_ids)={len(kept)} "
                    f"!= n_kept={entry['n_kept']}"
                )
            rej = entry.get("rejected_patient_ids")
            if not isinstance(rej, list):
                violations.append(f"cohorts[{name!r}].rejected_patient_ids must be a list")
            elif "n_rejected" in entry and len(rej) != entry["n_rejected"]:
                violations.append(
                    f"cohorts[{name!r}]: len(rejected_patient_ids)={len(rej)} "
                    f"!= n_rejected={entry['n_rejected']}"
                )
    if violations:
        joined = "\n  - ".join(violations)
        raise DedupDecisionSchemaError(f"decision.json {path} failed validation:\n  - {joined}")
    return payload


def load_dedup_decision(path: Path | str) -> dict[str, Any]:
    """Load + validate; alias for :func:`assert_dedup_decision_valid`."""
    return assert_dedup_decision_valid(path)


def build_allowlists(payload: dict[str, Any]) -> dict[str, set[str]]:
    """Convert a validated payload into per-cohort allow-list sets.

    The output is the in-memory structure passed to
    :class:`vena.model.fm.lightning.data.MultiCohortLatentDataModule`.
    """
    return {name: set(entry["kept_patient_ids"]) for name, entry in payload["cohorts"].items()}


def write_decision(path: Path | str, payload: dict[str, Any]) -> None:
    """Write the payload to ``path`` as pretty JSON (UTF-8)."""
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False))

"""Core frozen-dataclass types for the priors-validation preflight.

These types are the contract surface between the engine, the tests, and the
reporting layer. Every field is typed; everything is immutable so the
parallel runner cannot accidentally mutate shared state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vena.data.niigz import NiftiVolume

Severity = Literal["info", "warning", "error"]
PriorClearance = Literal["passed", "warning", "failed", "not_evaluated"]


@dataclass(frozen=True)
class SubjectMetadata:
    """Per-subject metadata. ``repeat_scan_id`` gates T5."""

    subject_id: str
    age: float | None = None
    sex: str | None = None
    scanner: str | None = None
    field_strength_t: float | None = None
    pathology: str | None = None
    who_grade: int | None = None
    repeat_scan_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubjectInputs:
    """All volumes + metadata needed to validate the priors for one subject.

    Optional fields are ``None`` when the modality / mask is absent on disk.
    The ``derived_priors`` dict is keyed by channel name (e.g. ``"cbf"``).
    """

    subject_id: str
    t1pre: NiftiVolume
    t1gd: NiftiVolume
    brain_mask: NiftiVolume
    parenchyma_mask: NiftiVolume | None
    tumour_mask: NiftiVolume | None
    cbf: NiftiVolume | None
    adc: NiftiVolume | None
    chi: NiftiVolume | None
    swan_mag: NiftiVolume | None
    derived_priors: dict[str, NiftiVolume]
    metadata: SubjectMetadata


@dataclass(frozen=True)
class TestOutcome:
    """One numerical assertion's outcome within a test class.

    Mirrors the spec §2.1 shape. ``metric_value`` and ``threshold`` are kept
    loosely typed (float | None | tuple) so subclasses can encode ranges,
    confidence intervals, or scalar values uniformly.
    """

    test_id: str
    subject_id: str
    prior_id: str | None
    roi_id: str | None
    metric_name: str
    metric_value: float | None
    threshold: float | tuple[float, float] | None
    passed: bool
    severity: Severity
    diagnostic: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    """All outcomes for one subject + roll-up flags."""

    subject_id: str
    outcomes: tuple[TestOutcome, ...]
    overall_passed: bool
    failed_priors: frozenset[str]
    aborted: bool = False
    abort_reason: str | None = None


@dataclass(frozen=True)
class AtlasSpec:
    """Pointer to one atlas asset with its provenance metadata."""

    atlas_id: str
    version: str
    path: Path
    sha256: str | None
    source_url: str | None
    citation_doi: str | None
    description: str = ""


@dataclass(frozen=True)
class CohortReport:
    """Aggregate report for one cohort run.

    ``per_test_pass_rate`` keys are the test_id strings; values are pass rates
    in ``[0, 1]`` or ``None`` when not applicable.
    ``per_prior_clearance`` maps channel name → spec §7.2 recommendation.
    """

    n_subjects: int
    n_subjects_applicable: int
    per_test_pass_rate: dict[str, float | None]
    per_prior_clearance: dict[str, PriorClearance]
    cohort_pass_rate_overall: float
    training_clearance: bool
    subjects: tuple[ValidationResult, ...]
    atlas_versions: dict[str, str]
    routine_version: str
    warnings: tuple[str, ...] = ()

"""Cohort deduplication preflight.

Builds per-cohort allow-lists from a corpus registry + the
BraTS-2021 ↔ TCIA mapping xlsx + a priority list. Emits a versioned
``decision.json`` consumed by ``routines.fm.train`` as a hard pre-flight gate.

Public entry points:

- :class:`CohortDedupConfig`, :class:`CohortDedupEngine` — orchestrator.
- :func:`parse_brats2021_mapping`, :class:`Brats2021Mapping` — xlsx parser.
- :func:`resolve`, :class:`CohortClaim`, :class:`ResolverOutput` — resolver.
- :func:`assert_dedup_decision_valid`, :func:`load_dedup_decision`,
  :func:`build_allowlists` — decision.json contract.
"""

from __future__ import annotations

from vena.preflight.cohort_dedup.decision import (
    DEDUP_DECISION_SCHEMA_VERSION,
    DedupDecisionSchemaError,
    assert_dedup_decision_valid,
    build_allowlists,
    load_dedup_decision,
    write_decision,
)
from vena.preflight.cohort_dedup.engine import CohortDedupConfig, CohortDedupEngine
from vena.preflight.cohort_dedup.resolver import (
    CohortClaim,
    CohortDedupResolverError,
    ResolvedOverlap,
    ResolverOutput,
    UnresolvableOverlap,
    resolve,
)
from vena.preflight.cohort_dedup.xlsx import (
    Brats2021Mapping,
    MappingRow,
    parse_brats2021_mapping,
)

__all__ = [
    "DEDUP_DECISION_SCHEMA_VERSION",
    "Brats2021Mapping",
    "CohortClaim",
    "CohortDedupConfig",
    "CohortDedupEngine",
    "CohortDedupResolverError",
    "DedupDecisionSchemaError",
    "MappingRow",
    "ResolvedOverlap",
    "ResolverOutput",
    "UnresolvableOverlap",
    "assert_dedup_decision_valid",
    "build_allowlists",
    "load_dedup_decision",
    "parse_brats2021_mapping",
    "resolve",
    "write_decision",
]

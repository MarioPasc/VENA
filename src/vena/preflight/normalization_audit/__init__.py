"""``normalization_audit`` preflight — V3 normalisation variant audit.

Audits seven intensity-normalisation variants against the MAISI-V2 VAE
encode→decode round-trip on UCSF-PDGM (n=30 main) + four smoke cohorts
(n=5 each). Selects the variant that preserves the T1c gadolinium-
enhancement tail while keeping the VAE in-distribution.

Background: `.claude/notes/review/2026-06-22_s1_v2_tumor_synthesis_failure_diagnosis.md`
§3.3 / §4 H4 showed that the production normalisation
``percentile_normalise(lo=0, hi=99.5, fg=True, clip=True)`` clips the top
0.5 % of T1c foreground voxels — i.e. the enhancement signal we are trying
to model — to 1.0. After this clip, ⟨|T1c − T1pre|⟩ in WT equals
⟨|T1c − T1pre|⟩ in non-WT brain (0.3837 = 0.3837), and in enhancing-only
(ET) tissue the contrast is *smaller* than non-WT brain (0.2384).

Spec: `.claude/notes/changes/2026-06-22_s1_v3_normalization_exploration.md`.

Public surface
--------------
* :class:`NormalizationVariant` — frozen dataclass with id + apply().
* :func:`get_variant_registry` — dict[str, NormalizationVariant].
* :func:`joint_modality_percentile_normalise` — V4 helper.
* :class:`NormalizationAuditConfig` — Pydantic config schema.
* :class:`NormalizationAuditEngine` — orchestrator.
* :class:`NormalizationAuditDecisionV1` — produced ``decision.json`` schema.
* :func:`assert_normalization_audit_decision_valid` — consumer validator.
"""

from __future__ import annotations

from .config import NormalizationAuditConfig
from .decision import (
    DECISION_PRODUCER,
    DECISION_SCHEMA_VERSION,
    NormalizationAuditDecisionV1,
    PerVariantMetrics,
    assert_normalization_audit_decision_valid,
    write_decision_json,
)
from .engine import NormalizationAuditEngine
from .joint import joint_modality_percentile_normalise
from .variants import NormalizationVariant, get_variant_registry, register_variant

__all__ = [
    "DECISION_PRODUCER",
    "DECISION_SCHEMA_VERSION",
    "NormalizationAuditConfig",
    "NormalizationAuditDecisionV1",
    "NormalizationAuditEngine",
    "NormalizationVariant",
    "PerVariantMetrics",
    "assert_normalization_audit_decision_valid",
    "get_variant_registry",
    "joint_modality_percentile_normalise",
    "register_variant",
    "write_decision_json",
]

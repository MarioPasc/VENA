"""``decision.json`` v1.0 contract for the V3 normalisation audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DECISION_SCHEMA_VERSION: str = "1.0.0"
DECISION_PRODUCER: str = "vena.preflight.normalization_audit:0.1.0"


class PerVariantMetrics(BaseModel):
    """Aggregate metrics for one variant on the main cohort."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant_id: str
    variant_version: str
    params: dict[str, Any]

    n_patients: int = Field(..., ge=0)

    # A — VAE round-trip on T1c (load-bearing region: ET).
    mae_whole: float
    mae_et: float
    mae_netc: float
    mae_ed: float
    mae_bnwt: float
    psnr_whole_db: float
    psnr_et_db: float | None = None  # may be NaN when no patient has ET voxels
    ssim_whole: float

    # B — Image-space signal preservation (load-bearing ratio: ET vs BNWT).
    image_mean_abs_diff_et: float
    image_mean_abs_diff_bnwt: float
    image_signal_ratio_et_over_bnwt: float

    # C — Latent-space signal preservation.
    latent_mean_abs_delta_et: float
    latent_mean_abs_delta_bnwt: float
    latent_signal_ratio_et_over_bnwt: float
    latent_mean_abs_t1c_et: float
    latent_mean_abs_t1pre_et: float

    # D — Distribution shape (per modality).
    kl_divergence_per_modality: dict[str, float]
    kl_divergence_max: float

    # Stratified C4/C5 (large-ET stratum).
    image_signal_ratio_large_et_stratum: float | None = None
    latent_signal_ratio_large_et_stratum: float | None = None
    n_patients_large_et: int = 0

    # Acceptance gate per criterion.
    passes_c1_mae_whole: bool
    passes_c2_mae_et: bool
    passes_c3_kl: bool
    passes_c4_image_signal: bool
    passes_c5_latent_signal: bool
    passes_c7_psnr_whole: bool
    passes_all: bool


class SmokeCohortVerdict(BaseModel):
    """Smoke-cohort verification result for the winner only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cohort: str
    n_patients: int
    mae_whole: float
    mae_et: float
    image_signal_ratio_et_over_bnwt: float
    passes: bool


class NormalizationAuditDecisionV1(BaseModel):
    """v1.0 contract emitted by the V3 normalisation audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = DECISION_SCHEMA_VERSION
    produced_at: str
    producer: str = DECISION_PRODUCER
    git_sha: str | None = None

    vae_checkpoint: str
    vae_checkpoint_sha256: str

    main_cohort: str
    n_patients_main: int = Field(..., ge=0)
    patient_seed: int

    variants_tested: list[str]
    metrics_per_variant: dict[str, PerVariantMetrics]

    # Acceptance gate.
    acceptance_thresholds: dict[str, float]

    # Winner.
    winner: str | None = Field(
        ...,
        description=(
            "Variant id that passes all criteria + smoke verification. "
            "``null`` when no variant qualifies (use V0 as fallback)."
        ),
    )
    winner_rationale: str
    fallback_used: bool

    # Smoke cohort outcomes (winner only).
    smoke_cohorts: list[SmokeCohortVerdict] = Field(default_factory=list)

    # Next action.
    next_action: Literal["re_encode_all_cohorts", "fall_back_to_v0", "manual_review"]


def assert_normalization_audit_decision_valid(path: Path) -> NormalizationAuditDecisionV1:
    """Load + validate the audit decision JSON. Raises ``pydantic.ValidationError``."""
    blob = json.loads(Path(path).read_text())
    return NormalizationAuditDecisionV1.model_validate(blob)


def write_decision_json(path: Path, decision: NormalizationAuditDecisionV1) -> None:
    """Write a validated decision payload (indent=2 for human review)."""
    path.write_text(json.dumps(decision.model_dump(mode="json"), indent=2))


__all__ = [
    "DECISION_PRODUCER",
    "DECISION_SCHEMA_VERSION",
    "NormalizationAuditDecisionV1",
    "PerVariantMetrics",
    "SmokeCohortVerdict",
    "assert_normalization_audit_decision_valid",
    "write_decision_json",
]

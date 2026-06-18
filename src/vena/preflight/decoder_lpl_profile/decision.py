"""``decision.json`` v1.0 contract for the ``decoder_lpl_profile`` preflight.

Schema bumps live in this module; the train engine asserts both the
``schema_version`` and the presence of every load-bearing field via
:func:`assert_decoder_lpl_decision_valid`. Keeping the schema isolated
makes consumer-side compatibility easy to audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DECISION_SCHEMA_VERSION: str = "1.0"
DECISION_PRODUCER: str = "vena.preflight.decoder_lpl_profile:1.0"


class _RegionRecipe(BaseModel):
    """§2.6 region-weighted variant recipe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha_wt: float = Field(..., description="WT region weight (per-region budget share).")
    alpha_notwt: float = Field(..., description="not-WT-in-brain region weight.")
    soft_region: bool = Field(
        ..., description="Whether to use the §2.6 soft-WT continuous variant."
    )
    per_cohort_overrides: dict[str, dict[str, float]] | None = Field(
        default=None,
        description=(
            "Per-cohort {alpha_wt, alpha_notwt} overrides when the §4.7c "
            "inter-cohort spread exceeds the configured threshold."
        ),
    )


class DecoderLplDecisionV1(BaseModel):
    """v1.0 contract emitted by the ``decoder_lpl_profile`` preflight.

    Consumed by ``routines.fm.train.engine._assert_preflight_gates`` when
    ``run.stage == 's3'`` (follow-up PR — this PR only emits the file).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = DECISION_SCHEMA_VERSION
    produced_at: str
    producer: str = DECISION_PRODUCER
    n_patients_run: int = Field(
        ...,
        ge=0,
        description="Total (patient, variant) cells the preflight processed.",
    )
    patients_per_cohort: dict[str, int] = Field(
        default_factory=dict,
        description="Per-cohort patient count actually visited.",
    )

    # Loss-recipe knobs measured by the preflight.
    A_recommended: list[int] = Field(
        ...,
        description=(
            "Decoder block indices selected for LPL readout. Drawn from the"
            " §4.2 error-concentration curve (top-2 by depth-weighted residual)."
        ),
    )
    w_l: dict[int, float] = Field(
        ...,
        description=(
            "Per-block depth weight measured from the §4.1 magnitude curve."
            " Keyed by block index; keys ⊆ A_recommended."
        ),
    )
    t_min: float = Field(
        ...,
        ge=0.0,
        lt=1.0,
        description="High-SNR gate; pinned at the §4.2 reliability knee.",
    )
    outlier_k: dict[int, float] = Field(
        ...,
        description="Per-block k·MAD outlier-mask threshold (Berrada §3.4).",
    )
    region_recipe: _RegionRecipe

    # Augmentation set.
    allowed_variants: list[str] = Field(
        ...,
        description=(
            "Augmentation variants that passed the §4.7b drift gate. The"
            " future S3 train YAML masks every absent variant from its"
            " ``variant_weights`` block."
        ),
    )
    v4_brain_mask_status: Literal["ok", "broken_drop_v4"] = Field(
        ...,
        description=(
            "Was the v4 brain-mask inflation (data-audit 2026-06-18) detected?"
            " ``broken_drop_v4`` forces v4 out of ``allowed_variants``."
        ),
    )

    @field_validator("w_l", mode="before")
    @classmethod
    def _coerce_int_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {int(k): float(vv) for k, vv in v.items()}
        return v

    @field_validator("outlier_k", mode="before")
    @classmethod
    def _coerce_int_keys_k(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {int(k): float(vv) for k, vv in v.items()}
        return v


def assert_decoder_lpl_decision_valid(path: Path) -> DecoderLplDecisionV1:
    """Load + validate a ``decoder_lpl_profile`` ``decision.json``.

    Returns the parsed Pydantic model so downstream code (the future
    ``_assert_preflight_gates`` extension) can read keys without
    re-validating. Raises :class:`pydantic.ValidationError` on schema
    mismatch.
    """
    blob = json.loads(Path(path).read_text())
    return DecoderLplDecisionV1.model_validate(blob)


def write_decision_json(path: Path, decision: DecoderLplDecisionV1) -> None:
    """Write a validated decision payload to disk (indent=2 for human review)."""
    path.write_text(json.dumps(decision.model_dump(mode="json"), indent=2))


__all__ = [
    "DECISION_PRODUCER",
    "DECISION_SCHEMA_VERSION",
    "DecoderLplDecisionV1",
    "assert_decoder_lpl_decision_valid",
    "write_decision_json",
]

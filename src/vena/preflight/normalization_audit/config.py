"""Pydantic config schema for the V3 normalisation audit preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SmokeCohortSpec(BaseModel):
    """A single smoke-verification cohort: path + sample count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Cohort id, e.g. 'BraTS-GLI'.")
    image_h5: Path = Field(..., description="Absolute path to image H5.")
    n_patients: int = Field(default=5, ge=1)


class NormalizationAuditConfig(BaseModel):
    """V3 normalisation audit configuration.

    YAML-driven; loaded via :meth:`from_yaml` in the routine's CLI.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # MAISI VAE.
    vae_checkpoint: Path = Field(..., description="MAISI-V2 VAE checkpoint path.")
    vae_arch_config: Path | None = Field(
        default=None,
        description=(
            "Optional MAISI VAE architecture JSON. When omitted the loader's "
            "default 4-channel MR config is used."
        ),
    )

    # Main cohort (UCSF-PDGM).
    main_cohort_name: str = Field(
        default="UCSF-PDGM",
        description="Main-sweep cohort id; written into decision.json.",
    )
    image_h5_main: Path = Field(..., description="Main-cohort image H5 path.")
    n_patients_main: int = Field(default=30, ge=1)
    patient_seed: int = Field(default=1337, description="Deterministic patient sampling seed.")
    patient_ids_main: list[str] | None = Field(
        default=None,
        description=(
            "Optional explicit list of patient ids to use instead of random "
            "sampling. When non-empty, ``n_patients_main`` is overridden."
        ),
    )

    # Smoke cohorts.
    smoke_cohorts: list[SmokeCohortSpec] = Field(
        default_factory=list,
        description=("Smoke-verification cohorts (winner only). Empty disables smoke."),
    )

    # Variants to test.
    variants_to_test: list[str] = Field(
        default_factory=lambda: ["V0", "V1", "V2", "V3", "V4", "V7", "V8"],
        description="Variant ids to sweep on the main cohort.",
    )

    # Modalities.
    modalities: list[str] = Field(
        default_factory=lambda: ["t1pre", "t1c", "t2", "flair"],
        description="Image-H5 modality keys to encode.",
    )
    target_modality: Literal["t1c"] = Field(
        default="t1c",
        description=(
            "Per-region metrics + signal ratios are computed against this "
            "modality (T1c is the FM target)."
        ),
    )

    # Execution.
    out_dir: Path = Field(..., description="Artifact root (timestamped subdir created inside).")
    device: str = Field(default="cuda:0")
    keep_intermediate_recons: bool = Field(
        default=False,
        description=(
            "When True, persists every (variant, patient, modality) decoded "
            "volume to disk. False keeps only per-patient metric rows + the "
            "per-variant aggregate."
        ),
    )

    # Acceptance criteria (overridable for sensitivity sweeps).
    c1_mae_whole_max: float = Field(default=0.010, gt=0.0, description="C1 threshold.")
    c2_mae_et_max: float = Field(default=0.015, gt=0.0, description="C2 threshold.")
    c3_kl_max_nats: float = Field(default=1.0, gt=0.0, description="C3 threshold.")
    c4_image_signal_ratio_min: float = Field(default=1.5, description="C4 threshold.")
    c5_latent_signal_ratio_min: float = Field(default=1.3, description="C5 threshold.")
    c7_psnr_whole_min_db: float = Field(default=35.0, description="C7 threshold (dB).")
    et_voxel_threshold_large: int = Field(
        default=10_000,
        description=(
            "ET voxel count above which a patient is in the 'large-ET' "
            "stratum used to evaluate C4/C5."
        ),
    )

    @field_validator("variants_to_test")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("variants_to_test must not be empty")
        return v

    @field_validator("modalities")
    @classmethod
    def _modalities_required(cls, v: list[str]) -> list[str]:
        required = {"t1pre", "t1c"}
        missing = required - set(v)
        if missing:
            raise ValueError(
                f"modalities must include {sorted(required)}; missing {sorted(missing)}"
            )
        return v

"""Configuration model for the ρ_S normalisation audit preflight.

One canonical percentile_upper is chosen and applied to BOTH synth and
real during metric computation.  This preflight determines which value
should be used in every downstream analysis.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class RhoSNormAuditConfig(BaseModel):
    """Frozen Pydantic config for the ρ_S normalisation audit.

    Parameters
    ----------
    inference_root :
        Root directory containing all inference shards
        (``picasso_shard_*/``).  Predictions are discovered recursively.
    frozen_csv_path :
        Path to the per-scan ``spatial_residual.csv`` from the pre-registered
        frozen sweep (2026-07-20T08-41-41Z).  Used to compute Δρ_S = P − frozen.
    output_root :
        Directory under which ``artifacts/preflights/rho_s_norm_audit/<UTC>/``
        is created.
    image_h5_map :
        Mapping from cohort name (e.g. ``"UCSF-PDGM"``) to the LOCAL image
        H5 path.  Required only for the P=99.95 normalisation path; not used
        for P=99.5.
    methods :
        Subset of method keys to audit.  ``null`` = all 16 pre-registered
        methods.
    cohorts :
        Subset of cohort names to include.  ``null`` = all Ring-A cohorts.
    percentiles :
        Percentile values to force for both synth and real.  Default
        ``[99.5, 99.95]``.
    condition :
        Spatial-residual condition to report (``"C-noT"`` = brain∖dilate(WT),
        the headline metric; ``"C-WB"`` = whole-brain).  Default ``"C-noT"``.
    scan_limit :
        Cap on the number of scans processed per (method, cohort) combination.
        ``null`` = unlimited.  Use ``10`` for the pilot, ``3`` for smoke.
    ring :
        Patient ring to restrict to.  Default ``"A"`` (247 patients, 321
        scans across 6 cohorts).
    log_level :
        Python logging level string.  Default ``"INFO"``.
    """

    inference_root: Path
    frozen_csv_path: Path
    output_root: Path
    image_h5_map: dict[str, Path] = Field(default_factory=dict)
    methods: list[str] | None = None
    cohorts: list[str] | None = None
    percentiles: list[float] = Field(default_factory=lambda: [99.5, 99.95])
    condition: str = "C-noT"
    scan_limit: int | None = None
    ring: str = "A"
    log_level: str = "INFO"

    model_config = {"frozen": True}

    @field_validator("percentiles")
    @classmethod
    def _check_percentiles(cls, v: list[float]) -> list[float]:
        for p in v:
            if p not in (99.5, 99.95):
                raise ValueError(f"Only percentiles 99.5 and 99.95 are supported; got {p}")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> RhoSNormAuditConfig:
        """Load config from a YAML file."""
        with open(path) as fh:
            blob = yaml.safe_load(fh)
        return cls.model_validate(blob)

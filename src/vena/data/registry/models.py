"""Pydantic models for the multi-cohort corpus registry.

The registry is a small JSON catalogue listing every cohort that participates
in a training run: its pathology, label system, role (train/val/test vs
test-only), whether it is longitudinal, the paths to its image- and
latent-domain H5 caches, and which modalities it carries. It is the single
input that the training config points at instead of one hard-coded latent H5,
so adding a cohort is a registry edit rather than a code change.

Splits live inside each cohort's H5 (patient-level, quota-based); the registry
only declares membership and roles. See ``.claude/rules/h5-design-principles.md``
and the project plan for the schema rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class RegistryError(Exception):
    """Raised when a corpus registry is structurally or referentially invalid."""


class CohortEntry(BaseModel):
    """One cohort's entry in the corpus registry.

    Attributes
    ----------
    name : str
        Cohort tag, must match the H5 ``cohort`` root attr (e.g. ``"UCSF-PDGM"``).
    pathology : str
        Coarse pathology label (e.g. ``"glioma"``).
    label_system : str
        Segmentation label convention (e.g. ``"BraTS2021"`` = {0,1,2,4},
        ``"BraTS2023"`` = {0,1,2,3}). Recorded for provenance; the whole-tumour
        mask is label-agnostic.
    role : {"cv", "test_only"}
        ``"cv"`` cohorts contribute to pooled train/val + test; ``"test_only"``
        cohorts are held out entirely for testing.
    longitudinal : bool
        Whether the cohort has multiple scans per patient (patient-level CSR).
    image_h5, latent_h5 : Path
        Absolute paths to the cohort's image- and latent-domain caches.
    n_patients, n_scans : int
        Counts (``n_scans >= n_patients``; equal for cross-sectional cohorts).
    modalities : list[str]
        Modality slugs present (e.g. ``["t1pre", "t1c", "t2", "flair"]``).
    has_swan : bool
        Whether SWAN/SWI is available (UCSF yes, BraTS no).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    pathology: str
    label_system: str
    role: Literal["cv", "test_only"]
    longitudinal: bool
    image_h5: Path
    latent_h5: Path
    n_patients: int
    n_scans: int
    modalities: list[str]
    has_swan: bool

    @field_validator("n_scans")
    @classmethod
    def _scans_ge_patients(cls, v: int, info) -> int:  # type: ignore[no-untyped-def]
        n_patients = info.data.get("n_patients")
        if n_patients is not None and v < n_patients:
            raise ValueError(f"n_scans ({v}) < n_patients ({n_patients})")
        return v


class CorpusRegistry(BaseModel):
    """Top-level corpus catalogue fed into the training config.

    Attributes
    ----------
    schema_version : str
        Registry-format version; bump on breaking changes.
    name : str
        Human-readable corpus name (e.g. ``"glioma_preop_v1"``).
    cohorts : list[CohortEntry]
        All participating cohorts; names must be unique.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "1.0.0"
    name: str
    cohorts: list[CohortEntry]

    @field_validator("cohorts")
    @classmethod
    def _unique_non_empty(cls, v: list[CohortEntry]) -> list[CohortEntry]:
        if not v:
            raise ValueError("corpus registry must list at least one cohort")
        seen: set[str] = set()
        for c in v:
            if c.name in seen:
                raise ValueError(f"duplicate cohort name in registry: {c.name!r}")
            seen.add(c.name)
        return v

    def by_name(self, name: str) -> CohortEntry:
        for c in self.cohorts:
            if c.name == name:
                return c
        raise RegistryError(f"no cohort named {name!r} in corpus {self.name!r}")

    def cv_cohorts(self) -> list[CohortEntry]:
        """Cohorts that contribute to pooled train/val (and their own test)."""
        return [c for c in self.cohorts if c.role == "cv"]

    def test_cohorts(self) -> list[CohortEntry]:
        """Cohorts whose patients are held out entirely for test."""
        return [c for c in self.cohorts if c.role == "test_only"]

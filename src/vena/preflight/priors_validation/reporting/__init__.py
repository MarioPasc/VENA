"""Reporting layer: per-subject PDF, cohort PDF, JSON, parquet."""

from __future__ import annotations

from .cohort_summary import write_cohort_outputs
from .per_subject import write_per_subject_json, write_per_subject_pdf

__all__ = ["write_cohort_outputs", "write_per_subject_json", "write_per_subject_pdf"]

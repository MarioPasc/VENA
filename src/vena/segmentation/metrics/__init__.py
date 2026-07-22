"""Evaluation metrics for vena.segmentation.

Task-15 symbols (overlap, calibration, gate) will be added here when that
task lands.  This module already exposes the task-40 visualisation helpers.
"""

from __future__ import annotations

from vena.segmentation.metrics.visualize import (
    PatientView,
    compute_mask_stats,
    compute_residual_energy_ratio,
    render_injection_sanity,
    render_latent_embedding,
    render_mask_qc,
    render_slice_montage,
)

__all__: list[str] = [
    "PatientView",
    "compute_mask_stats",
    "compute_residual_energy_ratio",
    "render_injection_sanity",
    "render_latent_embedding",
    "render_mask_qc",
    "render_slice_montage",
]

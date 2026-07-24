"""Evaluation metrics for vena.segmentation.

Exposes the task-40 visualisation helpers (PatientView, compute_mask_stats,
render_*) and the task-15 overlap / calibration / gate symbols.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Task-15: overlap, calibration, G-SEG gate, dual selection
# ---------------------------------------------------------------------------
from vena.segmentation.metrics.calibration import brier, classwise_ece, expected_calibration_error
from vena.segmentation.metrics.gate import GSegResult, ModelScore, check_gseg, select_ensemble
from vena.segmentation.metrics.overlap import average_hausdorff, dice, et_diagnostic

# ---------------------------------------------------------------------------
# Task-40: visualisation helpers (do NOT reorder or remove)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Task-17 (DEVELOPMENT): prediction panel for segmenter training callbacks
# ---------------------------------------------------------------------------
from vena.segmentation.metrics.visualize import (
    PanelRow,
    PatientView,
    compute_mask_stats,
    compute_residual_energy_ratio,
    render_injection_sanity,
    render_latent_embedding,
    render_mask_qc,
    render_prediction_panel,
    render_slice_montage,
    sort_panel_rows,
)

__all__: list[str] = [
    "GSegResult",
    "ModelScore",
    "PanelRow",
    "PatientView",
    "average_hausdorff",
    "brier",
    "check_gseg",
    "classwise_ece",
    "compute_mask_stats",
    "compute_residual_energy_ratio",
    "dice",
    "et_diagnostic",
    "expected_calibration_error",
    "render_injection_sanity",
    "render_latent_embedding",
    "render_mask_qc",
    "render_prediction_panel",
    "render_slice_montage",
    "select_ensemble",
    "sort_panel_rows",
]

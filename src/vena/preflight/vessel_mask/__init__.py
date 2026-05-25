"""Vessel-mask pre-flight: vesselness method QC on SWAN.

Library implementation of the routine ``routines/preflights/vessel_mask``.
Synthetic phantoms and analysis primitives are exposed here so they can be
imported by both the engine and the pytest suite without invoking the CLI.
"""

from __future__ import annotations

from .analysis import (
    PerPatientSweepRecord,
    PerTagSummary,
    ThresholdSweepResult,
    binary_fraction,
    connected_components_stats,
    dice,
    jaccard,
    otsu_threshold_brainmasked,
    pick_threshold_by_anatomical_fraction,
    skeleton_length,
    sweep_thresholds,
)
from .engine import (
    VesselMaskPreflightConfig,
    VesselMaskPreflightEngine,
    VesselMaskPreflightError,
)
from .synthetic import (
    cylinder_volume,
    parallel_cylinders_volume,
    rotated_cylinder_volume,
    smooth_cylinder_volume,
)

__all__ = [
    "PerPatientSweepRecord",
    "PerTagSummary",
    "ThresholdSweepResult",
    "VesselMaskPreflightConfig",
    "VesselMaskPreflightEngine",
    "VesselMaskPreflightError",
    "binary_fraction",
    "connected_components_stats",
    "cylinder_volume",
    "dice",
    "jaccard",
    "otsu_threshold_brainmasked",
    "parallel_cylinders_volume",
    "pick_threshold_by_anatomical_fraction",
    "rotated_cylinder_volume",
    "skeleton_length",
    "smooth_cylinder_volume",
    "sweep_thresholds",
]

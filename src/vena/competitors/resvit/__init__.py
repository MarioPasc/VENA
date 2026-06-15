"""VENA wrapper for the ResViT competitor (Dalmaz, Yurt, Çukur, IEEE TMI 2022).

Exports the deterministic 2D-slice datasets, the two-stage training entrypoint
``train_resvit``, and the per-patient inference helper ``run_inference``.
"""

from __future__ import annotations

from .dataset import (
    CohortImageSliceDataset,
    MultiCohortImageSliceDataset,
    UCSFPDGMSliceDataset,
)
from .inference import run_inference
from .runner import train_resvit

__all__ = [
    "CohortImageSliceDataset",
    "MultiCohortImageSliceDataset",
    "UCSFPDGMSliceDataset",
    "run_inference",
    "train_resvit",
]

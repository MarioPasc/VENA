"""Data loading and K-fold splits for vena.segmentation (task 14)."""

from __future__ import annotations

from vena.segmentation.data.augment import RandModalityDropout, build_augmentation
from vena.segmentation.data.dataset import SegImageDataset
from vena.segmentation.data.fm_splits import (
    CohortSplit,
    FmSplitResolution,
    resolve_fm_splits,
    write_splits_json,
)
from vena.segmentation.data.kfold import FoldPlan, build_fold_plan, oof_assignment

__all__ = [
    "CohortSplit",
    "FmSplitResolution",
    "FoldPlan",
    "RandModalityDropout",
    "SegImageDataset",
    "build_augmentation",
    "build_fold_plan",
    "oof_assignment",
    "resolve_fm_splits",
    "write_splits_json",
]

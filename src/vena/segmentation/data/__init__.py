"""Data loading and K-fold splits for vena.segmentation (task 14)."""

from __future__ import annotations

from vena.segmentation.data.augment import RandModalityDropout, build_augmentation
from vena.segmentation.data.dataset import SegImageDataset
from vena.segmentation.data.kfold import FoldPlan, build_fold_plan, oof_assignment

__all__ = [
    "FoldPlan",
    "RandModalityDropout",
    "SegImageDataset",
    "build_augmentation",
    "build_fold_plan",
    "oof_assignment",
]

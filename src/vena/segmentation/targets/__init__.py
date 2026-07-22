"""Soft target generation for vena.segmentation (task 12).

Public API
----------
harmonise_labels : Convert BraTS integer labels to boolean WT/NETC masks.
signed_distance  : Compute signed distance transform (per-component or geodesic).
soft_target      : sigmoid(SDT / sigma_vox) — soft probability from a binary mask.
make_soft_targets: Full pipeline: label → (2, H, W, D) float32 [WT, NETC].
"""

from __future__ import annotations

from vena.segmentation.targets.harmonise import harmonise_labels
from vena.segmentation.targets.sdt import signed_distance
from vena.segmentation.targets.soft_targets import make_soft_targets, soft_target

__all__ = [
    "harmonise_labels",
    "make_soft_targets",
    "signed_distance",
    "soft_target",
]

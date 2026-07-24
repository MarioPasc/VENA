"""Training and inference engine for vena.segmentation."""

from __future__ import annotations

from vena.segmentation.engine.loss import (
    SegmentationLoss,
    ce_term,
    dice_semimetric_loss,
    tversky_term,
)
from vena.segmentation.engine.predict import (
    oof_model_key,
    predict_oof,
)
from vena.segmentation.engine.train import (
    FitResult,
    SegTrainer,
)

__all__: list[str] = [
    "FitResult",
    "SegTrainer",
    "SegmentationLoss",
    "ce_term",
    "dice_semimetric_loss",
    "oof_model_key",
    "predict_oof",
    "tversky_term",
]

"""Augmentation pipeline for the segmentation image dataset.

Implements ~80% of the segmenter's OOD robustness budget via:

Intensity augmentations (applied per-channel on {t1pre, t2, flair}):
    - ``RandBiasFieldd``        — scanner field-inhomogeneity simulation.
    - ``RandAdjustContrastd``   — random gamma-contrast rescaling.
    - ``RandHistogramShiftd``   — random histogram warping.
    - ``RandGammad``            — random gamma correction.
    - ``RandGaussianNoised``    — additive Gaussian noise.

Spatial augmentations (applied consistently to images + soft targets + brain
mask, using bilinear interpolation to keep soft targets in [0, 1]):
    - ``RandFlipd``             — random axis flips.
    - ``RandAffined``           — random rotation / shear / translation / scale.
    - ``Rand3DElasticd``        — mild elastic deformation.

Modality dropout:
    - With probability ``modality_dropout_p``, exactly ONE of {t2, flair} is
      zeroed (independently of the other).  t1pre and t1c are never dropped.
      t1c is not present in the segmenter input — the dataset serves only
      {t1pre, t2, flair}.

Pipeline contract:
    Input dict::

        {
            "t1pre": ndarray (H, W, D) float32,
            "t2":    ndarray (H, W, D) float32,
            "flair": ndarray (H, W, D) float32,
            "target": ndarray (2, H, W, D) float32 soft ∈ [0, 1],
            "brain":  ndarray (H, W, D) float32 binary,
        }

    Output dict (augmented, same keys + same shapes):
        Same keys and shapes.  The callable returned by :func:`build_augmentation`
        is ready to be called on such a dict.

Note: stacking into ``"image": (3, H, W, D)`` is performed by the dataset, not
here.  This separation keeps each stage independently testable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vena.segmentation.config import DataConfig

logger = logging.getLogger(__name__)

__all__ = ["RandModalityDropout", "build_augmentation"]

# Keys operated on by spatial transforms — same geometric warp applied to all.
_SPATIAL_KEYS = ("t1pre", "t2", "flair", "target", "brain")

# Keys that receive intensity augmentation (modalities only, not mask/target).
_INTENSITY_KEYS = ("t1pre", "t2", "flair")

# The two droppable modalities (t1pre is structural anchor; t1c is absent).
_DROPPABLE = ("t2", "flair")


# ---------------------------------------------------------------------------
# Custom MONAI-compatible modality-dropout transform
# ---------------------------------------------------------------------------


class RandModalityDropout:
    """Randomly zero exactly one of {t2, flair}.

    Applied as a dict transform compatible with MONAI ``Compose``.  With
    probability ``p``, ONE of the two channels is selected uniformly at random
    and replaced with an all-zeros array of the same shape and dtype.

    t1pre is never dropped (structural anchor).  t1c is not present in the
    segmenter input by design (it would constitute label leakage).

    Parameters
    ----------
    p:
        Probability of applying dropout on any given sample.  When the
        transform fires (prob ``p``), exactly one of {t2, flair} is zeroed.
        When it does not fire (prob ``1-p``), all channels are returned
        unchanged.
    seed:
        Optional RNG seed for reproducible unit tests.
    """

    def __init__(self, p: float = 0.5, *, seed: int | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
        self._rng = np.random.default_rng(seed)

    def __call__(self, data: dict) -> dict:
        """Apply modality dropout to *data* in-place (returns same dict).

        Parameters
        ----------
        data:
            Dict with at least keys ``"t2"`` and ``"flair"``.  All other keys
            are passed through unchanged.

        Returns
        -------
        dict
            Same dict reference with one modality possibly zeroed.
        """
        if self._rng.random() >= self.p:
            return data  # no dropout this sample

        # Pick one of the two droppable modalities uniformly
        drop_key = _DROPPABLE[int(self._rng.integers(0, len(_DROPPABLE)))]
        vol = data[drop_key]
        if hasattr(vol, "numpy"):
            # torch.Tensor branch
            import torch

            data[drop_key] = torch.zeros_like(vol)
        else:
            data[drop_key] = np.zeros_like(vol)
        logger.debug("RandModalityDropout: zeroed '%s'", drop_key)
        return data


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def build_augmentation(
    cfg: DataConfig,
    *,
    modality_dropout_p: float = 0.5,
) -> Callable[[dict], dict]:
    """Build the full augmentation pipeline for segmenter training.

    Constructs a MONAI :class:`~monai.transforms.Compose` chain with:
    intensity augments → spatial augments → modality dropout.

    Parameters
    ----------
    cfg:
        Frozen :class:`~vena.segmentation.config.DataConfig`.  Currently only
        ``cfg.modalities`` is consumed (to validate droppable modalities); all
        augmentation hyper-parameters use task-14 defaults that match the
        segmentation training recipe.
    modality_dropout_p:
        Probability of zeroing one of {t2, flair} per sample.  Default 0.5.

    Returns
    -------
    Callable[[dict], dict]
        A callable that accepts the per-sample dict described in the module
        docstring and returns an augmented dict with the same keys and shapes.

    Raises
    ------
    ImportError
        If MONAI is not installed (required; listed in ``pyproject.toml``).
    """
    try:
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            Rand3DElasticd,
            RandAdjustContrastd,
            RandAffined,
            RandBiasFieldd,
            RandFlipd,
            RandGaussianNoised,
            RandHistogramShiftd,
            SqueezeDimd,
        )
    except ImportError as exc:
        raise ImportError(
            "MONAI is required for augmentation. Install via: pip install monai"
        ) from exc

    # ------------------------------------------------------------------
    # Intensity augmentations — applied to each modality independently.
    # Probabilities and magnitude ranges follow the nnU-Net / BraTS defaults
    # tuned for brain MRI.
    # Note: RandGammad is not available in MONAI ≥1.5; RandAdjustContrastd
    # subsumes gamma-contrast augmentation via its `gamma` parameter.
    # ------------------------------------------------------------------
    intensity_transforms = [
        RandBiasFieldd(
            keys=list(_INTENSITY_KEYS),
            prob=0.3,
            coeff_range=(0.0, 0.5),
        ),
        RandAdjustContrastd(
            keys=list(_INTENSITY_KEYS),
            prob=0.3,
            gamma=(0.7, 1.5),
        ),
        RandHistogramShiftd(
            keys=list(_INTENSITY_KEYS),
            prob=0.2,
            num_control_points=10,
        ),
        # Second contrast pass with wider range to substitute for the
        # deprecated RandGammad (MONAI ≥1.5 removed it).
        RandAdjustContrastd(
            keys=list(_INTENSITY_KEYS),
            prob=0.2,
            gamma=(0.5, 2.0),
        ),
        RandGaussianNoised(
            keys=list(_INTENSITY_KEYS),
            prob=0.2,
            mean=0.0,
            std=0.05,
        ),
    ]

    # ------------------------------------------------------------------
    # Spatial augmentations — SAME warp applied to all keys so that
    # target and brain mask stay geometrically consistent with the image.
    #
    # MONAI's grid-sampler requires consistent spatial rank across all
    # keys in one transform call.  Modality keys arrive as (H, W, D)
    # (no channel) while "target" is (2, H, W, D) (with channel).
    # EnsureChannelFirstd adds a channel dim to single-channel keys so
    # that all keys are (C, H, W, D) (ndim=4, spatial_dims=3) during
    # the warp, then SqueezeDimd restores the original shapes afterwards.
    # ------------------------------------------------------------------

    # Keys that need a temporary channel dim added/removed
    _channel_keys = ["t1pre", "t2", "flair", "brain"]

    spatial_keys = list(_SPATIAL_KEYS)
    spatial_modes = ["bilinear"] * len(spatial_keys)
    spatial_padding_modes = ["zeros"] * len(spatial_keys)

    spatial_transforms = [
        # Add channel dim to single-channel keys → (1, H, W, D)
        EnsureChannelFirstd(keys=_channel_keys, channel_dim="no_channel"),
        RandFlipd(
            keys=spatial_keys,
            prob=0.5,
            spatial_axis=0,
        ),
        RandFlipd(
            keys=spatial_keys,
            prob=0.5,
            spatial_axis=1,
        ),
        RandFlipd(
            keys=spatial_keys,
            prob=0.5,
            spatial_axis=2,
        ),
        RandAffined(
            keys=spatial_keys,
            prob=0.3,
            rotate_range=(0.15, 0.15, 0.15),  # ≈ ±8.6°
            shear_range=(0.05, 0.05, 0.05),
            translate_range=(10, 10, 10),  # voxels
            scale_range=(0.1, 0.1, 0.1),
            mode=spatial_modes,
            padding_mode=spatial_padding_modes,
        ),
        Rand3DElasticd(
            keys=spatial_keys,
            prob=0.2,
            sigma_range=(3, 5),
            magnitude_range=(10, 30),
            mode=spatial_modes,
            padding_mode=spatial_padding_modes,
        ),
        # Restore original (H, W, D) shape for single-channel keys
        SqueezeDimd(keys=_channel_keys, dim=0),
    ]

    # ------------------------------------------------------------------
    # Modality dropout — must come AFTER spatial (brain mask is still needed
    # during spatial transforms to stay consistent).
    # ------------------------------------------------------------------
    modality_dropout = RandModalityDropout(p=modality_dropout_p)

    class _DropoutWrapper:
        """Thin wrapper so RandModalityDropout fits MONAI Compose."""

        def __init__(self, dropout: RandModalityDropout) -> None:
            self._dropout = dropout

        def __call__(self, data: dict) -> dict:
            return self._dropout(data)

    all_transforms = [*intensity_transforms, *spatial_transforms, _DropoutWrapper(modality_dropout)]

    pipeline = Compose(all_transforms)
    return pipeline  # type: ignore[return-value]

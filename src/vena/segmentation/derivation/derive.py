"""Source-agnostic entrypoint: compute soft [WT, NETC] masks in MAISI latent space.

Both paths return a ``(2, 48, 56, 48)`` float32 Tensor in ``[0, 1]`` with the
same output contract so the caller can swap GT for predicted without changing
any downstream code (the **swap guarantee**).

GT path (``source="gt"``):
    ``harmonise_labels`` (inside :func:`make_soft_targets`) →
    ``make_soft_targets`` (SDT→sigmoid at image resolution) →
    :func:`pool_to_latent`.
    No temperature, no ensemble — the GT label is the oracle.

Predicted path (``source="predicted"``):
    Per-fold: :func:`apply_temperature` → :func:`pool_to_latent`.
    Then: :func:`ensemble_soft` (K-fold mean) → ``(2, 48, 56, 48)``.

Both paths produce identical shape, dtype, and value range.  Only the
group name written to disk and optional provenance attrs differ.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from vena.segmentation.derivation.ensemble import ensemble_soft
from vena.segmentation.derivation.pool import pool_to_latent
from vena.segmentation.derivation.temperature import ClassTemperatures, apply_temperature
from vena.segmentation.exceptions import SegDerivationError
from vena.segmentation.targets.soft_targets import make_soft_targets

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vena.common import CropPadSpec
    from vena.segmentation.config import DerivationConfig, TargetConfig

__all__ = ["derive_latent_soft_mask"]


def derive_latent_soft_mask(
    *,
    source: Literal["gt", "predicted"],
    label: NDArray | None = None,
    seg_prediction: Tensor | Sequence[Tensor] | None = None,
    temps: ClassTemperatures | None = None,
    image: NDArray | None = None,
    crop_spec: CropPadSpec | None = None,
    cfg: DerivationConfig,
    target_cfg: TargetConfig | None = None,
) -> Tensor:
    """Compute a soft [WT, NETC] probability map at the MAISI latent grid.

    Parameters
    ----------
    source : {"gt", "predicted"}
        Which derivation path to run.
    label : NDArray or None
        Integer label array ``(H, W, D)`` (BraTS-2021 or 2023 convention).
        Required when ``source="gt"``.
    seg_prediction : Tensor or Sequence[Tensor] or None
        Pre-sigmoid logits, shape ``(2, H, W, D)`` per fold, or a K-length
        sequence of such tensors.  Channel 0 = WT, channel 1 = NETC.
        Required when ``source="predicted"``.
    temps : ClassTemperatures or None
        Per-class temperature scalars ``(T_WT, T_NETC)``.  Required when
        ``source="predicted"``.
    image : NDArray or None
        Intensity volume ``(H, W, D)``.  Optional; passed to
        :func:`make_soft_targets` when ``cfg.netc_operator == "geodesic"``.
    crop_spec : CropPadSpec or None
        Per-scan brain-centred crop specification.  When provided, the
        native-space volume is cropped/padded to ``LATENT_CROP_BOX =
        (192, 224, 192)`` before avg-pooling.  When ``None``, the input
        must already be at crop-box spatial size.
    cfg : DerivationConfig
        Pooling settings: ``avg_pool_stride``, ``latent_grid``.
    target_cfg : TargetConfig or None
        Soft-target generation settings: ``sdt_sigma_vox``,
        ``netc_operator``, ``clip_vox``.  Required when ``source="gt"``.

    Returns
    -------
    Tensor
        Float32 soft probability map in ``[0, 1]``, shape
        ``(2, *cfg.latent_grid)`` = ``(2, 48, 56, 48)``.
        Channel 0 = WT (whole-tumour), channel 1 = NETC (necrotic core).
        Nesting is guaranteed: ``channel1 ≤ channel0`` elementwise.

    Raises
    ------
    SegDerivationError
        If required arguments for the chosen path are missing, if
        ``source`` is not recognised, or if the output grid deviates
        from ``cfg.latent_grid``.
    """
    if source == "gt":
        return _gt_path(
            label=label,
            image=image,
            crop_spec=crop_spec,
            cfg=cfg,
            target_cfg=target_cfg,
        )
    if source == "predicted":
        return _predicted_path(
            seg_prediction=seg_prediction,
            temps=temps,
            crop_spec=crop_spec,
            cfg=cfg,
        )
    raise SegDerivationError(f"unknown source {source!r}; expected 'gt' or 'predicted'")


# ---------------------------------------------------------------------------
# Private path implementations
# ---------------------------------------------------------------------------


def _gt_path(
    *,
    label: NDArray | None,
    image: NDArray | None,
    crop_spec: CropPadSpec | None,
    cfg: DerivationConfig,
    target_cfg: TargetConfig | None,
) -> Tensor:
    """GT oracle: SDT-soft at image res → avg-pool to latent grid."""
    if label is None:
        raise SegDerivationError("label is required when source='gt'")
    if target_cfg is None:
        raise SegDerivationError("target_cfg is required when source='gt'")

    # SDT → sigmoid at image resolution; returns numpy float32 (2, H, W, D).
    # harmonise_labels is called internally by make_soft_targets.
    soft_img: NDArray = make_soft_targets(label, target_cfg, image)

    # Convert to Tensor; pool_to_latent expects a Tensor.
    prob_img: Tensor = torch.from_numpy(soft_img)  # (2, H, W, D)

    # Optional crop to box + avg-pool → (2, 48, 56, 48).
    return pool_to_latent(prob_img, cfg, crop_spec=crop_spec)


def _predicted_path(
    *,
    seg_prediction: Tensor | Sequence[Tensor] | None,
    temps: ClassTemperatures | None,
    crop_spec: CropPadSpec | None,
    cfg: DerivationConfig,
) -> Tensor:
    """Predicted: temperature-scale + pool each fold, then K-fold mean."""
    if seg_prediction is None:
        raise SegDerivationError("seg_prediction is required when source='predicted'")
    if temps is None:
        raise SegDerivationError("temps is required when source='predicted'")

    # Normalise to a list of logit tensors; accept single Tensor as K=1.
    if isinstance(seg_prediction, Tensor):
        folds: list[Tensor] = [seg_prediction]
    else:
        folds = list(seg_prediction)

    if not folds:
        raise SegDerivationError("seg_prediction must contain at least one fold-logit tensor")

    # Per-fold pipeline: temperature-scale → sigmoid → pool to latent grid.
    pooled: list[Tensor] = []
    for fold_logits in folds:
        # apply_temperature returns sigmoid(logit / T) ∈ [0, 1].
        probs: Tensor = apply_temperature(fold_logits, temps)
        latent_probs: Tensor = pool_to_latent(probs, cfg, crop_spec=crop_spec)
        pooled.append(latent_probs)

    # K-fold mean (emit_variance=False → shape stays (2, *latent_grid)).
    return ensemble_soft(pooled, emit_variance=False)

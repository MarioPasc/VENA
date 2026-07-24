"""Out-of-fold ensemble prediction for the segmentation submodule.

Routes each patient to the correct fold model (FM-train patients) or the
all-train model (FM-val/test patients), asserts no leakage, then returns
image-resolution soft masks.  Pooling to the latent grid is a downstream
routine's responsibility — this module is free of H5 / latent concerns.

Channel semantics (output): channel-0 = TC, channel-1 = NETC.
Both channels carry independent sigmoid probabilities (nested, not exclusive).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from vena.segmentation.config import SegmentationConfig
from vena.segmentation.data.kfold import FoldPlan, oof_assignment
from vena.segmentation.exceptions import SegDataError
from vena.segmentation.models import get_segmentation_model

__all__ = ["load_seg_checkpoint", "oof_model_key", "predict_oof"]

logger = logging.getLogger(__name__)

# TTA flip axes in (B, C, H, W, D) format — spatial dims start at index 2.
_TTA_FLIP_DIMS: tuple[int, ...] = (2, 3, 4)
# original + 3 axis-flips = 4 views total (≥ _TTA_MIN_VIEWS).
_TTA_MIN_VIEWS: int = 2

# Sliding-window overlap fraction (matches training).
_SWI_OVERLAP: float = 0.25


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def oof_model_key(plan: FoldPlan, patient_id: str) -> int | Literal["all_train"]:
    """Return the OOF model key for *patient_id*.

    Thin wrapper around :func:`~vena.segmentation.data.kfold.oof_assignment`.

    Parameters
    ----------
    plan:
        The K-fold split plan.
    patient_id:
        Patient to look up.

    Returns
    -------
    int | Literal["all_train"]
        Fold index for FM-train patients; ``"all_train"`` for FM-val/test.

    Raises
    ------
    SegDataError
        If *patient_id* is not found in any partition of *plan*.
    """
    return oof_assignment(plan, patient_id)


# ---------------------------------------------------------------------------
# Internal helpers (module-level — not nested in loops)
# ---------------------------------------------------------------------------


def load_seg_checkpoint(
    cfg: SegmentationConfig,
    ckpt_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a :class:`SegTrainer` checkpoint into a fresh model instance.

    The checkpoint written by :meth:`~vena.segmentation.engine.train.SegTrainer.fit`
    embeds a ``"model_meta"`` dict that encodes the exact architecture
    (``model_name``, ``feature_size``, ``in_channels``, ``out_channels``,
    ``deep_supervision``).  When present, this metadata is used to reconstruct
    the model instead of relying on *cfg*, keeping save and load symmetric.
    Legacy checkpoints without ``"model_meta"`` fall back to *cfg.model*.

    Parameters
    ----------
    cfg:
        Segmentation config.  Used only when the checkpoint lacks ``"model_meta"``.
    ckpt_path:
        Path to a ``.pt`` checkpoint written by :class:`~vena.segmentation.engine.train.SegTrainer`.
    device:
        Target device.

    Returns
    -------
    nn.Module
        Model in eval mode on *device*, loaded with ``strict=True``.
    """
    from vena.segmentation.config import ModelConfig

    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    meta = state.get("model_meta")
    if meta is not None:
        # Reconstruct exact architecture from the checkpoint's embedded metadata
        model_cfg = ModelConfig(
            name=meta["model_name"],
            feature_size=meta.get("feature_size", cfg.model.feature_size),
            in_channels=meta.get("in_channels", cfg.model.in_channels),
            out_channels=meta.get("out_channels", cfg.model.out_channels),
            deep_supervision=meta.get("deep_supervision", cfg.model.deep_supervision),
        )
    else:
        # Legacy checkpoint without embedded metadata — trust the caller's cfg
        logger.warning(
            "Checkpoint %s lacks 'model_meta'; falling back to cfg.model for reconstruction.",
            ckpt_path,
        )
        model_cfg = cfg.model

    model = get_segmentation_model(model_cfg.name, model_cfg).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    logger.debug("Loaded checkpoint: %s (model=%s)", ckpt_path, model_cfg.name)
    return model


def _run_swi(
    model: nn.Module,
    images: Tensor,
    patch_size: tuple[int, int, int],
) -> Tensor:
    """Run sliding-window inference and return sigmoid soft probabilities.

    Parameters
    ----------
    model:
        Eval-mode segmentation model.
    images:
        Input tensor ``(B, C, H, W, D)`` already on the correct device.
    patch_size:
        Inference patch size; must match the training patch size.

    Returns
    -------
    Tensor
        Soft probabilities ``(B, 2, H, W, D)`` in ``[0, 1]``.
    """
    from monai.inferers import sliding_window_inference

    def _predictor(x: Tensor) -> Tensor:
        out = model(x)
        return out[0] if isinstance(out, tuple) else out

    with torch.no_grad():
        logits: Tensor = sliding_window_inference(
            inputs=images,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=_predictor,
            overlap=_SWI_OVERLAP,
        )
    return torch.sigmoid(logits)


def _apply_tta(
    model: nn.Module,
    images: Tensor,
    patch_size: tuple[int, int, int],
) -> Tensor:
    """Average predictions across flips along each spatial axis (TTA).

    Applies the original view plus one flip per spatial axis (dims 2, 3, 4),
    giving ``1 + len(_TTA_FLIP_DIMS) = 4`` views total (≥ :data:`_TTA_MIN_VIEWS`).
    The flipped prediction is un-flipped before averaging so all views are
    in the original coordinate frame.

    Parameters
    ----------
    model:
        Eval-mode segmentation model.
    images:
        Input tensor ``(B, C, H, W, D)``.
    patch_size:
        Inference patch size.

    Returns
    -------
    Tensor
        Averaged soft probabilities ``(B, 2, H, W, D)`` in ``[0, 1]``.
    """
    views: list[Tensor] = [_run_swi(model, images, patch_size)]

    for dim in _TTA_FLIP_DIMS:
        flipped = torch.flip(images, [dim])
        pred_flipped = _run_swi(model, flipped, patch_size)
        views.append(torch.flip(pred_flipped, [dim]))

    assert len(views) >= _TTA_MIN_VIEWS, (  # invariant guard
        f"TTA produced {len(views)} views; expected ≥ {_TTA_MIN_VIEWS}"
    )
    return torch.stack(views).mean(dim=0)


def _assert_no_leakage(
    plan: FoldPlan,
    routing: dict[str, int | Literal["all_train"]],
) -> None:
    """Raise :exc:`SegDataError` if any patient would be scored by a model that trained on it.

    Rules:

    * ``key == "all_train"``: the all-train model trains on ALL
      ``plan.fm_train_ids``; only FM-val/test patients (not in
      ``fm_train_ids``) may be routed here.
    * ``key == int(i)``: the fold-*i* model trains on
      ``fm_train_ids - folds[i]``; the patient must be in ``folds[i]``
      (i.e., it was held out) to be fairly predicted.

    Parameters
    ----------
    plan:
        The K-fold split plan.
    routing:
        ``{patient_id: oof_key}`` produced by :func:`oof_assignment`.

    Raises
    ------
    SegDataError
        On any detected leakage.
    """
    fm_train_set = set(plan.fm_train_ids)
    fold_sets: list[set[str]] = [set(f) for f in plan.folds]

    for pid, key in routing.items():
        if key == "all_train":
            if pid in fm_train_set:
                raise SegDataError(
                    f"Leakage: patient '{pid}' is in fm_train_ids but routed "
                    "to the all-train model (which trained on all fm_train_ids). "
                    "Only FM-val/test patients should be routed to all-train."
                )
        else:
            fold_idx = int(key)
            if pid not in fold_sets[fold_idx]:
                raise SegDataError(
                    f"Leakage: patient '{pid}' routed to fold-{fold_idx} model "
                    f"but is NOT in plan.folds[{fold_idx}]. "
                    "The fold-{fold_idx} model trained on this patient."
                )


# ---------------------------------------------------------------------------
# Public: predict_oof
# ---------------------------------------------------------------------------


def predict_oof(
    cfg: SegmentationConfig,
    ckpts: Mapping[int | str, Path],
    plan: FoldPlan,
    patient_ids: Sequence[str],
    *,
    tta: bool = False,
    dataset_factory: Callable[..., Dataset] | None = None,
    device: str | None = None,
) -> dict[str, Tensor]:
    """Predict soft masks for *patient_ids* using OOF fold models.

    Routes each patient to the correct checkpoint via
    :func:`~vena.segmentation.data.kfold.oof_assignment`, loads each
    checkpoint exactly once, and runs sliding-window inference at
    ``cfg.data.patch_size`` resolution.

    Parameters
    ----------
    cfg:
        Segmentation config.
    ckpts:
        ``{fold_index | "all_train": Path}`` mapping to checkpoint files.
    plan:
        The K-fold split plan used for routing.
    patient_ids:
        Patients to predict.
    tta:
        If ``True``, average :data:`_TTA_MIN_VIEWS` augmented views
        (original + axis-flips along each spatial dim).
    dataset_factory:
        Optional replacement for
        :class:`~vena.segmentation.data.dataset.SegImageDataset` for testing.
        Signature: ``(ids, cfg, *, augment, target_cfg) -> Dataset``.
        ``None`` uses the real dataset.
    device:
        Torch device string.  Resolution order: explicit *device* →
        ``cfg.run.device`` → ``"cuda"`` if available → ``"cpu"``.

    Returns
    -------
    dict[str, Tensor]
        ``{patient_id: Tensor (2, H, W, D) float32 ∈ [0, 1]}`` at **image
        resolution**.  Pooling to the latent grid is a downstream routine's
        responsibility.

    Raises
    ------
    SegDataError
        If any patient cannot be routed, any required checkpoint key is
        missing from *ckpts*, or leakage is detected before inference starts.
    """
    # ---- device -----------------------------------------------------------
    if device is not None:
        dev = torch.device(device)
    elif cfg.run is not None:
        dev = torch.device(cfg.run.device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # ---- route all patients (raises SegDataError for unknown IDs) ---------
    routing: dict[str, int | Literal["all_train"]] = {}
    for pid in patient_ids:
        routing[pid] = oof_assignment(plan, pid)

    # ---- leakage assert: before any I/O -----------------------------------
    _assert_no_leakage(plan, routing)

    # ---- verify all required checkpoint keys are provided -----------------
    required_keys: set[int | str] = set(routing.values())
    missing_keys = required_keys - set(ckpts.keys())
    if missing_keys:
        raise SegDataError(
            f"Missing checkpoints for OOF keys: {sorted(str(k) for k in missing_keys)}. "
            f"Available: {sorted(str(k) for k in ckpts.keys())}"
        )

    # ---- group patients by OOF key (one checkpoint load per key) ----------
    key_to_pids: dict[int | str, list[str]] = {}
    for pid, key in routing.items():
        key_to_pids.setdefault(key, []).append(pid)

    patch_size = cfg.data.patch_size
    results: dict[str, Tensor] = {}

    for key, pids in key_to_pids.items():
        ckpt_path = ckpts[key]
        logger.info(
            "predict_oof: key=%s | ckpt=%s | n_patients=%d | tta=%s",
            key,
            ckpt_path,
            len(pids),
            tta,
        )
        model = load_seg_checkpoint(cfg, ckpt_path, dev)

        # ---- dataset for this group ---------------------------------------
        if dataset_factory is not None:
            ds = dataset_factory(
                ids=pids,
                cfg=cfg.data,
                augment=False,
                target_cfg=cfg.targets,
            )
        else:
            from vena.segmentation.data.dataset import SegImageDataset

            ds = SegImageDataset(
                ids=pids,
                cfg=cfg.data,
                augment=False,
                target_cfg=cfg.targets,
            )

        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        for batch in loader:
            images: Tensor = batch["image"].to(dev)
            raw_pids = batch["patient_id"]
            batch_pids: list[str] = raw_pids if isinstance(raw_pids, list) else [raw_pids]

            with torch.no_grad():
                if tta:
                    pred = _apply_tta(model, images, patch_size)
                else:
                    pred = _run_swi(model, images, patch_size)

            for b_idx, pid in enumerate(batch_pids):
                results[pid] = pred[b_idx].cpu()

        logger.info("predict_oof: key=%s done", key)

    return results

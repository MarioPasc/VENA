"""Segmentation trainer: one model per invocation, K-fold or all-train mode.

Channel semantics: channel-0 = TC (tumour core = NETC + ET), channel-1 = NETC.
Independent sigmoids per channel (not softmax) — TC and NETC are nested,
not mutually exclusive.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from vena.segmentation.config import SegmentationConfig
from vena.segmentation.data.kfold import FoldPlan
from vena.segmentation.engine.loss import SegmentationLoss
from vena.segmentation.exceptions import SegDataError
from vena.segmentation.metrics.calibration import brier, classwise_ece
from vena.segmentation.metrics.overlap import dice, et_diagnostic
from vena.segmentation.models import get_segmentation_model

if TYPE_CHECKING:
    pass

__all__ = ["FitResult", "SegTrainer"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (no magic numbers in logic)
# ---------------------------------------------------------------------------

_GRAD_CLIP_MAX_NORM: float = 1.0  # standard for 3-D segmentation models
_COSINE_LR_ETA_MIN_FRAC: float = 1e-3  # eta_min = lr * this fraction
_SCORE_EPS: float = 1e-8  # denominator guard for dual harmonic mean
_SWI_OVERLAP: float = 0.25  # sliding-window overlap fraction


# ---------------------------------------------------------------------------
# CSV writer — FM convention: freeze header on first write, never sparse rows
# ---------------------------------------------------------------------------


class _CSVWriter:
    """Append-only CSV writer that emits the header exactly once.

    Parameters
    ----------
    path:
        Output CSV path.
    columns:
        Column names written as the header on the first :meth:`write` call.
    """

    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self._path = path
        self._columns: tuple[str, ...] = tuple(columns)
        self._initialized: bool = False

    def write(self, row: dict[str, Any]) -> None:
        """Append one fully-populated row (missing keys → empty string)."""
        mode = "a" if self._initialized else "w"
        with self._path.open(mode, newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._columns)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow({k: row.get(k, "") for k in self._columns})


# ---------------------------------------------------------------------------
# Selection-metric helpers (always higher = better after this transform)
# ---------------------------------------------------------------------------


def _compute_val_score(dice_mean: float, brier_mean: float, mode: str) -> float:
    """Map per-epoch val metrics to a unified scalar where higher is better.

    Parameters
    ----------
    dice_mean:
        Mean Dice across both channels: (Dice_TC + Dice_NETC) / 2.
    brier_mean:
        Mean Brier score across both channels: (Brier_TC + Brier_NETC) / 2.
    mode:
        ``"dice"``, ``"brier"`` (negated so higher = better), or ``"dual"``
        (harmonic mean of Dice and 1 − Brier).

    Returns
    -------
    float
        Unified score in roughly [0, 1] where higher is always better.
    """
    if mode == "dice":
        return dice_mean
    if mode == "brier":
        return 1.0 - brier_mean
    if mode == "dual":
        a = dice_mean
        b = 1.0 - brier_mean
        return 2.0 * a * b / (a + b + _SCORE_EPS)
    raise ValueError(f"Unknown selection_metric '{mode}'")


# ---------------------------------------------------------------------------
# Tumour-aware patch extraction (training only)
# ---------------------------------------------------------------------------

# Positive/negative voxel sampling ratio for RandCropByPosNegLabeld.
# pos=1, neg=1 → ~50 % chance per sample of centring on a TC voxel, which is
# critical because TC occupies only ~0.2–0.8 % of the brain volume on UCSF-PDGM.
_CROP_POS: int = 1
_CROP_NEG: int = 1


def _build_tumour_crop_transform(patch_size: tuple[int, int, int]) -> Any:
    """Build a MONAI ``RandCropByPosNegLabeld`` transform (created once, reused).

    The transform crops ``"image"``, ``"target"``, and ``"brain"`` (if present)
    jointly, guided by the hard TC label stored under key ``"label_tc"``.

    Parameters
    ----------
    patch_size:
        Target ``(pH, pW, pD)``.

    Returns
    -------
    Any
        Callable ``(dict) -> list[dict]`` (num_samples=1).
    """
    from monai.transforms import RandCropByPosNegLabeld

    return RandCropByPosNegLabeld(
        keys=["image", "target", "brain"],
        label_key="label_tc",
        spatial_size=patch_size,
        pos=_CROP_POS,
        neg=_CROP_NEG,
        num_samples=1,
        allow_smaller=True,
    )


def _apply_tumour_crop(
    batch: dict[str, Any],
    patch_size: tuple[int, int, int],
    transform: Any,
) -> dict[str, Any]:
    """Apply tumour-aware random crop to a collated batch.

    For each sample in the batch the transform is applied independently; results
    are re-stacked along the batch dimension.

    When the volume is already ≤ patch_size in all spatial dims the transform
    is a no-op (MONAI pads to patch_size with ``allow_smaller=True``; callers
    that use genuinely smaller test volumes accept the padding).

    Parameters
    ----------
    batch:
        Collated DataLoader batch with 5-D tensors ``(B, C, H, W, D)``.
        Must contain ``"image"`` and ``"target"``; ``"brain"`` is optional.
    patch_size:
        Crop target — must match the transform's ``spatial_size``.
    transform:
        Pre-built callable from :func:`_build_tumour_crop_transform`.

    Returns
    -------
    dict[str, Any]
        Same dict with ``"image"``, ``"target"``, and (if present) ``"brain"``
        replaced by cropped 5-D tensors of shape ``(B, C, *patch_size)``.
    """
    images = batch["image"]  # (B, C, H, W, D)
    targets = batch["target"]  # (B, 2, H, W, D)
    brains: Tensor | None = batch.get("brain")  # (B, 1, H, W, D) or None

    cropped_images: list[Tensor] = []
    cropped_targets: list[Tensor] = []
    cropped_brains: list[Tensor] = []

    for b in range(images.shape[0]):
        sample: dict[str, Tensor] = {
            "image": images[b],  # (C, H, W, D)
            "target": targets[b],  # (2, H, W, D)
            # Hard TC mask used by the pos/neg sampler
            "label_tc": (targets[b, 0:1] > 0.5).float(),  # (1, H, W, D)
            # brain: use ones if not in batch so the key is always present
            "brain": brains[b]
            if brains is not None
            else torch.ones(1, *images.shape[-3:], dtype=images.dtype),
        }
        result_list = transform(sample)
        result: dict[str, Tensor] = result_list[0] if isinstance(result_list, list) else result_list
        cropped_images.append(result["image"])
        cropped_targets.append(result["target"])
        cropped_brains.append(result["brain"])

    out: dict[str, Any] = {k: v for k, v in batch.items() if k not in ("image", "target", "brain")}
    out["image"] = torch.stack(cropped_images)
    out["target"] = torch.stack(cropped_targets)
    if brains is not None:
        out["brain"] = torch.stack(cropped_brains)
    return out


# ---------------------------------------------------------------------------
# Auxiliary helpers (module-level — not nested in loops)
# ---------------------------------------------------------------------------


def _get_main_logits(outputs: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Return the primary logit tensor from a possibly-tuple model output."""
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _pin_viz_patients(
    val_ids: Sequence[str],
    explicit_ids: tuple[str, ...] | None,
    n: int,
    seed: int,
) -> tuple[str, ...]:
    """Select and pin the visualization patient IDs at construction time.

    Parameters
    ----------
    val_ids:
        Available validation patient IDs.
    explicit_ids:
        If set, use these directly (filtered to those present in val_ids).
    n:
        Number of patients to select when *explicit_ids* is None.
    seed:
        RNG seed for deterministic selection.

    Returns
    -------
    tuple[str, ...]
        Pinned patient IDs (subset of val_ids, identical across every epoch).
    """
    if explicit_ids is not None:
        available = set(val_ids)
        pinned = tuple(pid for pid in explicit_ids if pid in available)
        return pinned if pinned else tuple(sorted(val_ids))[:n]

    sorted_ids = sorted(val_ids)
    rng = np.random.default_rng(seed)
    k = min(n, len(sorted_ids))
    if k == 0:
        return ()
    chosen_idx = rng.choice(len(sorted_ids), size=k, replace=False)
    return tuple(sorted_ids[i] for i in sorted(chosen_idx))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """Summary of a completed :class:`SegTrainer` run.

    Attributes
    ----------
    run_dir:
        Root directory for all run artifacts.
    checkpoint:
        Path to the best-scoring checkpoint (``best.pt``).
    best_epoch:
        Zero-indexed epoch where the best validation score was achieved.
    best_score:
        Best unified selection-metric value (higher = better).
    initial_train_loss:
        Mean training loss over the first epoch.
    final_train_loss:
        Mean training loss over the last completed epoch.
    history:
        One dict per validation event containing all val metrics plus
        ``"epoch"``.
    """

    run_dir: Path
    checkpoint: Path
    best_epoch: int
    best_score: float
    initial_train_loss: float
    final_train_loss: float
    history: tuple[dict[str, float], ...]


class SegTrainer:
    """Trains one segmentation model (fold-i or all-train) against a FoldPlan.

    One instance = one model.  K-fold ensemble training runs K+1 instances
    with different *fold* arguments (typically as separate SLURM array jobs).

    Parameters
    ----------
    cfg:
        Frozen top-level segmentation config.
    fold:
        ``int`` in ``[0, plan.k)`` → train on all FM-train patients minus
        ``plan.folds[fold]``, validate on ``plan.folds[fold]`` (true OOF).
        ``"all_train"`` → train on all ``plan.fm_train_ids``, hold out a
        deterministic seeded slice of size ``cfg.train.calibration_split_frac``
        for early-stopping monitoring only.

        **No true OOF validation set exists for the all-train model.**
        Its purpose is to predict the FM val/test patients that are never
        seen by any fold model.
    plan:
        The K-fold split plan constructed externally (e.g. via
        :func:`~vena.segmentation.data.kfold.build_fold_plan`).
    run_dir:
        Directory for all run artifacts (checkpoints, logs, metrics, figures).
        Created if absent.
    patient_to_scans:
        Optional patient → scan-id expansion for longitudinal cohorts
        (e.g. LUMIERE: 91 patients / 599 scans).  ``None`` means identity.
    dataset_factory:
        Replaces :class:`~vena.segmentation.data.dataset.SegImageDataset`
        for testing.  Signature::

            factory(ids, cfg, *, augment, target_cfg) -> Dataset

        ``None`` uses the real :class:`~vena.segmentation.data.dataset.SegImageDataset`.
    viz_renderer:
        Injected panel-rendering callable.  ``None`` triggers a lazy import of
        ``vena.segmentation.metrics.visualize.render_prediction_panel``; if
        that import fails (Lane B not yet merged), viz is silently skipped.
    """

    def __init__(
        self,
        cfg: SegmentationConfig,
        fold: int | Literal["all_train"],
        *,
        plan: FoldPlan,
        run_dir: Path,
        patient_to_scans: Mapping[str, Sequence[str]] | None = None,
        dataset_factory: Callable[..., Dataset] | None = None,
        viz_renderer: Callable[..., Path] | None = None,
    ) -> None:
        self._cfg = cfg
        self._fold = fold
        self._plan = plan
        self._run_dir = Path(run_dir)
        self._patient_to_scans = patient_to_scans
        self._dataset_factory = dataset_factory
        self._viz_renderer = viz_renderer

        # Resolve train/val patient IDs (cheap — no I/O)
        self._train_patient_ids, self._val_patient_ids = self._resolve_patient_split()

        # Expand patients → scans for longitudinal cohorts
        self._train_scan_ids: tuple[str, ...] = self._expand_scans(self._train_patient_ids)
        self._val_scan_ids: tuple[str, ...] = self._expand_scans(self._val_patient_ids)

        # Effective seed: run.seed > run.seed=None→cfg.seed > cfg.seed
        self._effective_seed: int = (
            (cfg.run.seed if cfg.run.seed is not None else cfg.seed)
            if cfg.run is not None
            else cfg.seed
        )

        # Pin viz patients once at construction (same panel subjects every epoch)
        self._viz_patient_ids: tuple[str, ...] = _pin_viz_patients(
            val_ids=self._val_patient_ids,
            explicit_ids=cfg.viz.patient_ids,
            n=cfg.viz.n_patients,
            seed=self._effective_seed,
        )

    # ------------------------------------------------------------------
    # Private helpers: split resolution
    # ------------------------------------------------------------------

    def _resolve_patient_split(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Compute (train_ids, val_ids) for this fold.

        Returns
        -------
        tuple[tuple[str, ...], tuple[str, ...]]
            ``(train_ids, val_ids)`` — patient-level (not scan-level).

        Raises
        ------
        SegDataError
            If *fold* is an int outside ``[0, plan.k)``.
        """
        plan = self._plan
        cfg = self._cfg

        if isinstance(self._fold, int):
            fold_idx = self._fold
            if fold_idx < 0 or fold_idx >= plan.k:
                raise SegDataError(f"fold={fold_idx} out of range [0, {plan.k})")
            val_ids = plan.folds[fold_idx]
            val_set = set(val_ids)
            train_ids = tuple(pid for pid in plan.fm_train_ids if pid not in val_set)
            return train_ids, val_ids

        # all_train: hold out calibration_split_frac for monitoring
        all_ids = sorted(plan.fm_train_ids)
        n_val = max(1, round(len(all_ids) * cfg.train.calibration_split_frac))
        rng = np.random.default_rng(cfg.data.fold_seed)
        perm = rng.permutation(len(all_ids)).tolist()
        val_pos = set(perm[:n_val])
        val_ids_l = [all_ids[i] for i in range(len(all_ids)) if i in val_pos]
        train_ids_l = [all_ids[i] for i in range(len(all_ids)) if i not in val_pos]
        return tuple(train_ids_l), tuple(val_ids_l)

    def _expand_scans(self, patient_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Expand patient IDs → scan IDs via *patient_to_scans*.  Identity if None."""
        if self._patient_to_scans is None:
            return patient_ids
        expanded: list[str] = []
        for pid in patient_ids:
            scans = self._patient_to_scans.get(pid)
            if scans:
                expanded.extend(scans)
            else:
                expanded.append(pid)
        return tuple(expanded)

    # ------------------------------------------------------------------
    # Private helpers: dataset
    # ------------------------------------------------------------------

    def _build_dataset(self, scan_ids: Sequence[str], *, augment: bool) -> Dataset:
        """Construct a dataset for the given scan IDs."""
        cfg = self._cfg
        if self._dataset_factory is not None:
            return self._dataset_factory(
                ids=scan_ids,
                cfg=cfg.data,
                augment=augment,
                target_cfg=cfg.targets,
            )
        from vena.segmentation.data.dataset import SegImageDataset

        return SegImageDataset(
            ids=scan_ids,
            cfg=cfg.data,
            augment=augment,
            target_cfg=cfg.targets,
        )

    # ------------------------------------------------------------------
    # Private helpers: viz
    # ------------------------------------------------------------------

    def _resolve_renderer(self) -> Callable[..., Path] | None:
        """Resolve the panel renderer; return None if unavailable."""
        if self._viz_renderer is not None:
            return self._viz_renderer
        try:
            from vena.segmentation.metrics.visualize import (  # type: ignore[attr-defined]
                render_prediction_panel,
            )

            return render_prediction_panel
        except (ImportError, AttributeError) as exc:
            logger.warning("render_prediction_panel unavailable (%s); viz disabled.", exc)
            return None

    def _try_render_panel(
        self,
        renderer: Callable[..., Path] | None,
        val_preds: dict[str, Tensor],
        val_targets: dict[str, Tensor],
        per_patient_scores: dict[str, float],
        epoch: int,
        fig_dir: Path,
    ) -> None:
        """Render the qualitative prediction panel.  Any error → WARNING, no abort.

        Parameters
        ----------
        renderer:
            Resolved renderer callable (may be None → no-op).
        val_preds:
            Per-patient soft predictions ``(2, H, W, D)``.
        val_targets:
            Per-patient soft GT targets ``(2, H, W, D)``.
        per_patient_scores:
            Per-patient unified selection-metric values.
        epoch:
            Current training epoch (used in the output filename).
        fig_dir:
            Directory for figure output.
        """
        if renderer is None:
            return
        cfg = self._cfg
        try:
            from vena.segmentation.metrics.visualize import PanelRow  # type: ignore[attr-defined]
        except (ImportError, AttributeError) as exc:
            logger.warning("PanelRow unavailable (%s); skipping viz.", exc)
            return

        rows = []
        for pid in self._viz_patient_ids:
            if pid not in val_preds:
                continue
            pred = val_preds[pid]  # (2, H, W, D)
            tgt = val_targets[pid]  # (2, H, W, D)
            try:
                rows.append(
                    PanelRow(
                        patient_id=pid,
                        anatomy=tgt[0].cpu().numpy(),  # TC soft as anatomy proxy
                        gt_hard=(tgt > 0.5).cpu().numpy().astype(np.uint8),
                        pred_soft=pred.cpu().numpy(),
                        metric=per_patient_scores.get(pid, float("nan")),
                        metric_name=cfg.metrics.selection_metric,
                    )
                )
            except Exception as row_exc:
                logger.warning("PanelRow build failed for %s: %s", pid, row_exc)

        if not rows:
            return

        out_path = fig_dir / f"epoch_{epoch:03d}.png"
        try:
            renderer(
                rows,
                out_path,
                n_cols=cfg.viz.n_cols,
                title=f"Fold {self._fold} — Epoch {epoch}",
                gt_alpha=cfg.viz.gt_alpha,
                soft_alpha=cfg.viz.soft_alpha,
            )
            logger.info("Viz panel saved: %s", out_path)
        except Exception as render_exc:
            logger.warning("Viz rendering failed at epoch %d: %s", epoch, render_exc)

    # ------------------------------------------------------------------
    # Private helpers: validation pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_val(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        patch_size: tuple[int, int, int],
    ) -> tuple[dict[str, float], dict[str, Tensor], dict[str, Tensor]]:
        """Evaluate on the validation set.

        Parameters
        ----------
        model:
            Segmentation model (placed in eval mode by this method).
        val_loader:
            DataLoader over the validation set (batch_size=1, no augmentation).
        device:
            Target device.
        use_amp:
            Whether AMP autocast is enabled.
        amp_dtype:
            AMP precision dtype (bfloat16 or float16).
        patch_size:
            Inference patch size for sliding-window inference.

        Returns
        -------
        tuple
            ``(metrics, preds, targets)`` where:

            * ``metrics`` — aggregate metric dict (all val_* keys).
            * ``preds`` — ``{patient_id: (2,H,W,D) float Tensor}`` soft probabilities.
            * ``targets`` — ``{patient_id: (2,H,W,D) float Tensor}`` soft targets.
        """
        from monai.inferers import sliding_window_inference

        model.eval()

        # Build predictor closure (AMP inside — @no_grad wraps the whole method)
        dev_type = device.type

        def _predictor(x: Tensor) -> Tensor:
            with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)
            return _get_main_logits(out)

        dice_tc_list: list[float] = []
        dice_netc_list: list[float] = []
        brier_tc_list: list[float] = []
        brier_netc_list: list[float] = []
        ece_tc_list: list[float] = []
        ece_netc_list: list[float] = []
        et_dice_list: list[float] = []
        et_soft_list: list[float] = []

        preds: dict[str, Tensor] = {}
        targets: dict[str, Tensor] = {}

        for batch in val_loader:
            images: Tensor = batch["image"].to(device)  # (B, C, H, W, D)
            target: Tensor = batch["target"].to(device)  # (B, 2, H, W, D)

            # DataLoader collates strings into a list
            raw_pids = batch["patient_id"]
            patient_ids: list[str] = raw_pids if isinstance(raw_pids, list) else [raw_pids]

            pred_logits: Tensor = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=_predictor,
                overlap=_SWI_OVERLAP,
            )
            pred_soft: Tensor = torch.sigmoid(pred_logits)

            for b_idx, pid in enumerate(patient_ids):
                ps = pred_soft[b_idx].detach()  # (2, H, W, D)
                tgt = target[b_idx].detach()  # (2, H, W, D)
                tgt_hard = (tgt > 0.5).float()

                d_tc = dice(ps[0], tgt_hard[0])
                d_netc = dice(ps[1], tgt_hard[1])
                br = brier(ps, tgt_hard)
                ece = classwise_ece(ps, tgt_hard)
                etd = et_diagnostic(ps, tgt)

                dice_tc_list.append(d_tc)
                dice_netc_list.append(d_netc)
                brier_tc_list.append(br["tc"])
                brier_netc_list.append(br["netc"])
                ece_tc_list.append(ece["tc"])
                ece_netc_list.append(ece["netc"])
                et_dice_list.append(etd["et_dice"])
                et_soft_list.append(etd["mean_et_soft"])

                preds[pid] = ps.cpu()
                targets[pid] = tgt.cpu()

        n = max(len(dice_tc_list), 1)
        dice_tc_m = sum(dice_tc_list) / n
        dice_netc_m = sum(dice_netc_list) / n
        dice_m = (dice_tc_m + dice_netc_m) / 2.0
        brier_tc_m = sum(brier_tc_list) / n
        brier_netc_m = sum(brier_netc_list) / n
        brier_m = (brier_tc_m + brier_netc_m) / 2.0

        score = _compute_val_score(dice_m, brier_m, self._cfg.metrics.selection_metric)

        metrics: dict[str, float] = {
            "val_dice_tc": dice_tc_m,
            "val_dice_netc": dice_netc_m,
            "val_dice_mean": dice_m,
            "val_brier_tc": brier_tc_m,
            "val_brier_netc": brier_netc_m,
            "val_brier_mean": brier_m,
            "val_score": score,
            "val_ece_tc": sum(ece_tc_list) / n,
            "val_ece_netc": sum(ece_netc_list) / n,
            "val_et_dice": sum(et_dice_list) / n,
            "val_mean_et_soft": sum(et_soft_list) / n,
        }
        return metrics, preds, targets

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(self) -> FitResult:
        """Train the segmentation model and return a :class:`FitResult`.

        Writes into ``self._run_dir``:

        * ``checkpoints/best.pt`` — best-scoring checkpoint (model state dict +
          epoch + score).
        * ``checkpoints/last.pt`` — final-epoch checkpoint.
        * ``logs/train.log`` — plain :class:`logging.FileHandler` capturing all
          log messages from this process during training (FM convention).
        * ``metrics/train_step.csv`` — per-optimiser-step loss and LR; header
          frozen on first write so every row has the same columns.
        * ``metrics/train_epoch.csv`` — per-epoch mean training loss and LR.
        * ``metrics/val_epoch.csv`` — per-validation-event val metrics.
        * ``fold_plan.json`` — serialised :class:`FoldPlan` for provenance.
        * ``figures/epoch_NNN.png`` — qualitative prediction panels (when viz
          is enabled).

        .. note::
            No ``temperatures.json`` is written.  Temperature scaling was
            dropped in iter-9 decision Q5 (2026-07-23).  Calibration is
            *measured* (ECE / Brier in ``val_epoch.csv``) and never corrected.

        Returns
        -------
        FitResult
            Summary of the completed training run.
        """
        cfg = self._cfg
        run_dir = self._run_dir

        # ---- directory layout ------------------------------------------
        ckpt_dir = run_dir / "checkpoints"
        log_dir = run_dir / "logs"
        metrics_dir = run_dir / "metrics"
        fig_dir = run_dir / "figures"
        for d in (ckpt_dir, log_dir, metrics_dir, fig_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---- file handler (FM convention: self-contained run log) -------
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(fh)

        try:
            return self._fit_inner(cfg, run_dir, ckpt_dir, metrics_dir, fig_dir)
        finally:
            root_logger.removeHandler(fh)
            fh.close()

    def _fit_inner(
        self,
        cfg: SegmentationConfig,
        run_dir: Path,
        ckpt_dir: Path,
        metrics_dir: Path,
        fig_dir: Path,
    ) -> FitResult:
        """Inner training loop (separated to ensure FileHandler cleanup)."""
        # ---- provenance -------------------------------------------------
        (run_dir / "fold_plan.json").write_text(
            json.dumps(self._plan.to_dict(), indent=2), encoding="utf-8"
        )

        # ---- device / AMP -----------------------------------------------
        if cfg.run is not None:
            device_str = cfg.run.device
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
        device = torch.device(device_str)
        dev_type = "cuda" if device.type == "cuda" else "cpu"
        use_amp = cfg.train.amp and dev_type == "cuda"
        # float16 fallback when bf16 not supported (Volta GPUs, CPU runs always use fp32 via use_amp=False)
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        # torch.amp.GradScaler: first positional arg is `device` string (torch ≥ 2.6 convention).
        # enabled=False makes it a no-op on CPU, so device_str can safely be "cpu".
        scaler = torch.amp.GradScaler(device_str, enabled=use_amp)

        # ---- seeding ----------------------------------------------------
        seed = self._effective_seed
        torch.manual_seed(seed)

        # ---- model ------------------------------------------------------
        model = get_segmentation_model(cfg.model.name, cfg.model).to(device)
        if cfg.run is not None and cfg.run.resume_from is not None:
            state = torch.load(cfg.run.resume_from, map_location=device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            logger.info("Resumed from %s", cfg.run.resume_from)

        # ---- loss -------------------------------------------------------
        criterion = SegmentationLoss(cfg.loss)

        # ---- optimiser + cosine LR -------------------------------------
        optimizer = AdamW(model.parameters(), lr=cfg.train.lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.train.max_epochs, 1),
            eta_min=cfg.train.lr * _COSINE_LR_ETA_MIN_FRAC,
        )

        # ---- datasets / dataloaders ------------------------------------
        train_ds = self._build_dataset(self._train_scan_ids, augment=True)
        val_ds = self._build_dataset(self._val_scan_ids, augment=False)

        # persistent_workers avoids worker re-spawn overhead each epoch
        # (skip when num_workers=0 to avoid the "persistent_workers requires
        #  num_workers > 0" constraint).
        _pw = cfg.data.num_workers > 0
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=(dev_type == "cuda"),
            drop_last=False,
            persistent_workers=_pw,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=(dev_type == "cuda"),
            persistent_workers=_pw,
        )

        # ---- tumour-aware crop transform (created once, reused per batch) ---
        crop_transform = _build_tumour_crop_transform(cfg.data.patch_size)

        # ---- CSV writers (headers frozen up front) ----------------------
        step_csv = _CSVWriter(
            metrics_dir / "train_step.csv",
            ["epoch", "global_step", "loss", "lr"],
        )
        epoch_csv = _CSVWriter(
            metrics_dir / "train_epoch.csv",
            # data_wait_s / step_s let us size SLURM --time for Picasso array.
            # data_wait_s: wall time spent waiting for DataLoader batches.
            # step_s: wall time spent on forward + backward + optimiser.
            ["epoch", "loss_mean", "lr", "data_wait_s", "step_s"],
        )
        val_columns = [
            "epoch",
            "val_dice_tc",
            "val_dice_netc",
            "val_dice_mean",
            "val_brier_tc",
            "val_brier_netc",
            "val_brier_mean",
            "val_score",
            "val_ece_tc",
            "val_ece_netc",
            "val_et_dice",
            "val_mean_et_soft",
        ]
        val_csv = _CSVWriter(metrics_dir / "val_epoch.csv", val_columns)

        # ---- viz renderer (lazy import, graceful fallback) -------------
        renderer: Callable[..., Path] | None = None
        if cfg.viz.enabled:
            renderer = self._resolve_renderer()

        # ---- model metadata (embedded in every checkpoint for symmetric load) ---
        # predict_oof._load_model uses these to reconstruct the exact model
        # from the checkpoint alone, without trusting the caller's cfg.model.
        _model_meta: dict[str, Any] = {
            "model_name": cfg.model.name,
            "feature_size": cfg.model.feature_size,
            "in_channels": cfg.model.in_channels,
            "out_channels": cfg.model.out_channels,
            "deep_supervision": cfg.model.deep_supervision,
        }

        # ---- training state --------------------------------------------
        patch_size = cfg.data.patch_size
        metric_mode = cfg.metrics.selection_metric
        best_score: float = -math.inf
        best_epoch: int = 0
        patience_count: int = 0
        global_step: int = 0
        history: list[dict[str, float]] = []
        initial_train_loss: float | None = None
        final_train_loss: float = float("nan")

        autocast_ctx = torch.amp.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp)

        logger.info(
            "SegTrainer: fold=%s device=%s amp=%s seed=%d "
            "train_scans=%d val_scans=%d max_epochs=%d",
            self._fold,
            device,
            use_amp,
            seed,
            len(self._train_scan_ids),
            len(self._val_scan_ids),
            cfg.train.max_epochs,
        )

        for epoch in range(cfg.train.max_epochs):
            # ---- train one epoch ----------------------------------------
            model.train()
            epoch_losses: list[float] = []
            epoch_data_wait: float = 0.0
            epoch_step_time: float = 0.0

            batch_t0 = time.perf_counter()
            for batch in train_loader:
                data_wait = time.perf_counter() - batch_t0
                epoch_data_wait += data_wait

                step_t0 = time.perf_counter()

                # Tumour-aware crop: applies jointly to image, target, brain.
                # Validation uses sliding-window on the full volume so metrics
                # are whole-volume numbers comparable to the G-SEG gate.
                cropped = _apply_tumour_crop(batch, patch_size, crop_transform)
                images: Tensor = cropped["image"].to(device)
                target: Tensor = cropped["target"].to(device)

                optimizer.zero_grad()

                with autocast_ctx:
                    outputs = model(images)
                    loss: Tensor = criterion(outputs, target)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()

                epoch_step_time += time.perf_counter() - step_t0

                loss_val = float(loss.detach().cpu())
                epoch_losses.append(loss_val)
                step_csv.write(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": round(loss_val, 6),
                        "lr": round(optimizer.param_groups[0]["lr"], 8),
                    }
                )
                global_step += 1
                batch_t0 = time.perf_counter()

            scheduler.step()

            epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            final_train_loss = epoch_loss
            if initial_train_loss is None:
                initial_train_loss = epoch_loss

            epoch_csv.write(
                {
                    "epoch": epoch,
                    "loss_mean": round(epoch_loss, 6),
                    "lr": round(optimizer.param_groups[0]["lr"], 8),
                    "data_wait_s": round(epoch_data_wait, 3),
                    "step_s": round(epoch_step_time, 3),
                }
            )
            logger.debug("Epoch %d | train_loss=%.4f", epoch, epoch_loss)

            # ---- validation (every val_every_epochs + epoch 0) ----------
            is_val_epoch = epoch % cfg.train.val_every_epochs == 0
            if not is_val_epoch:
                continue

            val_metrics, val_preds, val_targets = self._run_val(
                model, val_loader, device, use_amp, amp_dtype, patch_size
            )
            val_row: dict[str, float] = {"epoch": float(epoch), **val_metrics}
            val_csv.write(val_row)
            history.append(val_row)

            score = val_metrics["val_score"]
            logger.info(
                "Epoch %d val | dice_mean=%.4f brier_mean=%.4f score=%.4f",
                epoch,
                val_metrics["val_dice_mean"],
                val_metrics["val_brier_mean"],
                score,
            )

            # ---- viz panel (epoch 0 + every viz.every_epochs) -----------
            is_viz_epoch = epoch == 0 or epoch % cfg.viz.every_epochs == 0
            if cfg.viz.enabled and is_viz_epoch:
                per_patient_scores: dict[str, float] = {}
                for pid in self._viz_patient_ids:
                    if pid in val_preds and pid in val_targets:
                        ps = val_preds[pid]
                        tgt = val_targets[pid]
                        tgt_h = (tgt > 0.5).float()
                        d_tc = dice(ps[0], tgt_h[0])
                        d_netc = dice(ps[1], tgt_h[1])
                        br = brier(ps, tgt_h)
                        d_m = (d_tc + d_netc) / 2.0
                        br_m = (br["tc"] + br["netc"]) / 2.0
                        per_patient_scores[pid] = _compute_val_score(d_m, br_m, metric_mode)
                self._try_render_panel(
                    renderer, val_preds, val_targets, per_patient_scores, epoch, fig_dir
                )

            # ---- checkpoint + early stopping ----------------------------
            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience_count = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "best_score": best_score,
                        "model_state_dict": model.state_dict(),
                        "model_meta": _model_meta,
                    },
                    ckpt_dir / "best.pt",
                )
                logger.info("New best at epoch %d (score=%.4f)", epoch, best_score)
            else:
                patience_count += 1
                if patience_count >= cfg.train.early_stop_patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d elapsed)",
                        epoch,
                        cfg.train.early_stop_patience,
                    )
                    break

        # ---- last checkpoint -------------------------------------------
        torch.save(
            {
                "epoch": cfg.train.max_epochs - 1,
                "model_state_dict": model.state_dict(),
                "model_meta": _model_meta,
            },
            ckpt_dir / "last.pt",
        )

        best_ckpt = ckpt_dir / "best.pt"
        if not best_ckpt.exists():
            # No val epoch ran (max_epochs < val_every_epochs)
            best_ckpt = ckpt_dir / "last.pt"
            best_epoch = 0
            best_score = float("nan")

        return FitResult(
            run_dir=run_dir,
            checkpoint=best_ckpt,
            best_epoch=best_epoch,
            best_score=best_score,
            initial_train_loss=float(initial_train_loss)
            if initial_train_loss is not None
            else float("nan"),
            final_train_loss=final_train_loss,
            history=tuple(history),
        )

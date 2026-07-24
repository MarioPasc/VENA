"""Thin engine for the segmentation training routine (task 18).

Responsibilities
----------------
1. Parse + validate YAML config via :class:`~vena.segmentation.config.SegmentationConfig`.
2. Resolve FM patient splits for all registry cohorts.
3. Build the K-fold plan over the FM train split.
4. Write ``splits.json`` (per-cohort patient lists — explicit user requirement).
5. Instantiate :class:`~vena.segmentation.engine.train.SegTrainer` and call ``fit()``.
6. After training, run a per-cohort evaluation pass to produce the G-SEG table.
7. Run :func:`~vena.segmentation.metrics.gate.check_gseg` and write ``decision.json``.
8. Log the sentinel lines ``seg-train completed`` and ``RUN_DIR=<path>``.

Design constraints (preflight-pattern.md)
------------------------------------------
- No heavy work at import time: no CUDA, no checkpoint load, no H5 open at module scope.
- ``Engine.run() -> Path`` returns the run directory.
- One positional YAML argument; all other config from the file.
- **No ``temperatures.json`` is written** — temperature scaling was dropped in
  iter-9 decision Q5 (2026-07-23).

Channel semantics: channel 0 = TC (NETC+ET, edema excluded), channel 1 = NETC.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1.0.0"
_PRODUCER = "routines.segmentation.train:1.0.0"

# Ring-B cohorts that must appear in the G-SEG table (test_only role).
_RING_B_COHORT_NAMES = frozenset({"BraTS-Africa-Glioma", "BraTS-Africa-Other", "BraTS-PED"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso_utc() -> str:
    """Return the current UTC time as a compact ISO-8601 string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _resolve_git_sha() -> str:
    """Return a 8-char short git SHA, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str | None:
    """Compute SHA-256 hex digest of a file; return None if file absent."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _fold_label(fold: int | str) -> str:
    """Convert fold to string for use in run_id."""
    return "all_train" if fold == "all_train" else str(fold)


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------


class SegTrainEngine:
    """Orchestrate one segmentation training run.

    Parameters
    ----------
    cfg:
        Frozen top-level segmentation configuration.  Produced by
        :meth:`~vena.segmentation.config.SegmentationConfig.from_yaml`.
    """

    def __init__(self, cfg: Any) -> None:
        # cfg is SegmentationConfig; typed Any to avoid import at module scope.
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Execute the training routine and return the run directory.

        Returns
        -------
        Path
            Absolute path to the run directory that was created.

        Raises
        ------
        ValueError
            If ``cfg.run`` is None (run section required for a real training job).
        routines.segmentation.train.exceptions.SegTrainError
            If any unrecoverable step fails (wrapped from underlying exceptions).
        """
        from vena.segmentation.config import SegmentationConfig  # type: ignore[import]

        cfg: SegmentationConfig = self._cfg

        if cfg.run is None:
            raise ValueError("cfg.run is None — a RunConfig section is required for engine.run().")

        # ---- logging bootstrap (before file handler) -------------------
        logging.basicConfig(
            level=getattr(logging, cfg.run.log_level.upper(), logging.INFO),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

        # ---- provenance ------------------------------------------------
        produced_at = _now_iso_utc()
        git_sha = _resolve_git_sha()
        fold_str = _fold_label(cfg.run.fold)
        run_id = f"{produced_at}_seg_{cfg.run.tag}_fold{fold_str}_{git_sha}"

        run_dir = Path(cfg.run.experiments_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # ---- attach file handler (FM convention: self-contained run) ---
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(fh)

        try:
            return self._run_inner(cfg, run_id, run_dir, produced_at, git_sha)
        finally:
            root_logger.removeHandler(fh)
            fh.close()

    # ------------------------------------------------------------------
    # Private: inner run (separated so FileHandler is always cleaned up)
    # ------------------------------------------------------------------

    def _run_inner(
        self,
        cfg: Any,
        run_id: str,
        run_dir: Path,
        produced_at: str,
        git_sha: str,
    ) -> Path:
        """Inner training logic — all I/O and model operations live here."""
        import torch

        from vena.segmentation.data.fm_splits import (  # type: ignore[import]
            resolve_fm_splits,
            write_splits_json,
        )
        from vena.segmentation.data.kfold import build_fold_plan  # type: ignore[import]
        from vena.segmentation.engine.train import SegTrainer  # type: ignore[import]

        # TF32 matmul precision (FM convention; ~10% step-time gain on A100)
        torch.set_float32_matmul_precision("high")

        # ---- seed -------------------------------------------------------
        seed = cfg.run.seed if cfg.run.seed is not None else cfg.seed
        torch.manual_seed(seed)

        logger.info("run_id=%s fold=%s seed=%d", run_id, cfg.run.fold, seed)

        # ---- resolve FM splits -----------------------------------------
        logger.info("Resolving FM splits from registry: %s", cfg.data.corpus_registry)
        resolution = resolve_fm_splits(cfg.data)

        # ---- build K-fold plan -----------------------------------------
        fm_splits = resolution.fm_splits()
        plan = build_fold_plan(
            cfg.data,
            fm_splits,
            dedup_duplicates=None,
            cohort_labels=resolution.patient_to_cohort,
        )
        logger.info(
            "Fold plan: k=%d  fm_train=%d  fm_val=%d  fm_test=%d",
            plan.k,
            len(plan.fm_train_ids),
            len(plan.fm_val_ids),
            len(plan.fm_test_ids),
        )

        # ---- write splits.json -----------------------------------------
        splits_path = run_dir / "splits.json"
        write_splits_json(splits_path, resolution, plan, extra={"run_id": run_id})
        logger.info("splits.json written: %s", splits_path)

        # ---- persist resolved config -----------------------------------
        _write_resolved_config(run_dir, cfg)

        # ---- run training ----------------------------------------------
        trainer = SegTrainer(
            cfg,
            cfg.run.fold,
            plan=plan,
            run_dir=run_dir,
            patient_to_scans=resolution.patient_to_scans,
        )
        logger.info("Starting SegTrainer.fit() for fold=%s", cfg.run.fold)
        result = trainer.fit()
        logger.info(
            "Training complete: best_epoch=%d best_score=%.4f",
            result.best_epoch,
            result.best_score,
        )

        # ---- per-cohort evaluation (G-SEG table) -----------------------
        logger.info("Starting per-cohort evaluation for decision.json ...")
        dice_by_cohort = _evaluate_per_cohort(
            cfg=cfg,
            result=result,
            resolution=resolution,
            plan=plan,
        )

        # ---- G-SEG gate -------------------------------------------------
        # Cohorts whose inference failed carry null metrics and a non-"ok"
        # status.  Passing None to float() inside check_gseg would raise
        # TypeError.  Separate null-metric cohorts first; they are a
        # *missing-data* failure distinct from "scored below threshold".
        from vena.segmentation.metrics.gate import (  # type: ignore[import]
            GSegResult,
            check_gseg,
        )

        gseg_input: dict[str, dict[str, float]] = {}
        missing_cohorts: list[tuple[str, str, Any]] = []

        for _cname, _mets in dice_by_cohort.items():
            if _mets.get("status", "ok") != "ok":
                missing_cohorts.append((_cname, "missing-data", _mets.get("status", "unknown")))
            else:
                # Strip non-numeric keys (status, n_evaluated) before gate
                gseg_input[_cname] = {
                    k: v
                    for k, v in _mets.items()
                    if k not in ("status", "n_evaluated") and isinstance(v, float)
                }

        gseg_result = check_gseg(gseg_input, cfg.metrics)

        if missing_cohorts:
            # Third element is a string (failure description), not a float —
            # this is intentional: it makes null-metric failures visually
            # distinct from numeric under-threshold failures in decision.json.
            combined: list[Any] = list(gseg_result.failures) + missing_cohorts
            gseg_result = GSegResult(
                passed=False,
                per_cohort=gseg_result.per_cohort,
                failures=combined,
            )
            logger.warning(
                "G-SEG gate FAILED — %d cohort(s) with null metrics: %s",
                len(missing_cohorts),
                [t[0] for t in missing_cohorts],
            )

        if gseg_result.passed:
            logger.info("G-SEG gate PASSED")
        else:
            logger.warning("G-SEG gate FAILED: %s", gseg_result.failures)

        # ---- write decision.json ---------------------------------------
        decision_path = _write_decision_json(
            run_dir=run_dir,
            cfg=cfg,
            run_id=run_id,
            produced_at=produced_at,
            git_sha=git_sha,
            resolution=resolution,
            plan=plan,
            result=result,
            gseg_result=gseg_result,
            dice_by_cohort=dice_by_cohort,
        )
        logger.info("decision.json written: %s", decision_path)

        # ---- sentinel lines (loginexa smoke greps for these) -----------
        logger.info("seg-train completed")
        logger.info("RUN_DIR=%s", run_dir.resolve())
        print("seg-train completed")
        print(f"RUN_DIR={run_dir.resolve()}")

        return run_dir.resolve()


# ---------------------------------------------------------------------------
# Per-cohort evaluation
# ---------------------------------------------------------------------------


def _evaluate_per_cohort(
    *,
    cfg: Any,
    result: Any,
    resolution: Any,
    plan: Any,
) -> dict[str, dict[str, float]]:
    """Build per-cohort metric dict suitable for ``check_gseg``.

    For **CV cohorts**, runs sliding-window inference on the OOF-held-out
    patients (those in ``plan.folds[fold]`` for fold models, or the
    ``calibration_split_frac`` slice for the all-train model).

    For **test_only cohorts** (Ring-B OOD), runs inference on all test patients.

    When H5 access fails or a cohort has no assigned patients, returns
    placeholder ``0.0`` values so the decision.json is always structurally
    complete.

    Parameters
    ----------
    cfg:
        Frozen :class:`~vena.segmentation.config.SegmentationConfig`.
    result:
        :class:`~vena.segmentation.engine.train.FitResult` from trainer.
    resolution:
        :class:`~vena.segmentation.data.fm_splits.FmSplitResolution`.
    plan:
        :class:`~vena.segmentation.data.kfold.FoldPlan`.

    Returns
    -------
    dict[str, dict[str, float]]
        Outer key = cohort name.  Inner keys: ``"tc"``, ``"netc"``
        (Dice for tumour cohorts) or ``"tc_volume"`` (for healthy cohorts).
        Additional keys stored for the decision.json but not consumed by
        ``check_gseg``: ``"tc_ahd"``, ``"netc_ahd"``, ``"tc_ece"``,
        ``"netc_ece"``, ``"tc_brier"``, ``"netc_brier"``,
        ``"et_dice"``, ``"mean_et_soft"``.
    """

    dice_by_cohort: dict[str, dict[str, float]] = {}

    # Identify which patients are the "evaluation" set for this fold
    fold = cfg.run.fold
    if fold == "all_train":
        # all-train model: no true OOF set; use val/test patients from the
        # FM split (those the all-train model was specifically trained to predict)
        oof_patient_ids: frozenset[str] = frozenset(plan.fm_val_ids) | frozenset(plan.fm_test_ids)
    else:
        # fold-i model: OOF patients are those in plan.folds[fold]
        oof_patient_ids = frozenset(plan.folds[fold])

    # patient → cohort mapping for stratification
    patient_to_cohort: Mapping[str, str] = resolution.patient_to_cohort

    for cs in resolution.per_cohort:
        cohort_name: str = cs.name

        # Determine evaluation patient set for this cohort
        if cs.role == "cv":
            # CV: evaluate on the OOF-held-out patients that belong to this cohort
            eval_patients = tuple(
                pid for pid in oof_patient_ids if patient_to_cohort.get(pid) == cohort_name
            )
        else:
            # test_only: evaluate on all test patients
            eval_patients = cs.test_patients

        metrics = _infer_cohort_metrics(
            cfg=cfg,
            best_ckpt=result.checkpoint,
            patient_ids=eval_patients,
            resolution=resolution,
            cohort_name=cohort_name,
            cs=cs,
        )
        dice_by_cohort[cohort_name] = metrics
        _tc = metrics.get("tc")
        _netc = metrics.get("netc")
        logger.info(
            "Cohort %s (%s): status=%s n=%d tc_dice=%s netc_dice=%s",
            cohort_name,
            cs.role,
            metrics.get("status", "ok"),
            len(eval_patients),
            f"{_tc:.3f}" if isinstance(_tc, float) else "null",
            f"{_netc:.3f}" if isinstance(_netc, float) else "null",
        )

    return dice_by_cohort


def _null_metrics(status: str, n_evaluated: int = 0) -> dict[str, float | None]:
    """Return a null-filled metric dict for cohorts where inference could not run.

    Parameters
    ----------
    status:
        Human-readable failure description, e.g.
        ``"error: OSError: /path/to/h5 not found"``.
    n_evaluated:
        Number of scans processed before failure (usually 0).

    Notes
    -----
    **Why null instead of 0.0?**  ``0.0`` Dice is a valid measurement (model
    predicted nothing correctly).  ``null`` is unambiguous: inference did not
    run.  Downstream consumers must not treat unmeasured cohorts as zero-scoring.
    """
    return {
        "status": status,
        "n_evaluated": n_evaluated,
        "tc": None,
        "netc": None,
        "tc_ahd": None,
        "netc_ahd": None,
        "tc_ece": None,
        "netc_ece": None,
        "tc_brier": None,
        "netc_brier": None,
        "et_dice": None,
        "mean_et_soft": None,
    }


def _infer_cohort_metrics(
    *,
    cfg: Any,
    best_ckpt: Path,
    patient_ids: tuple[str, ...],
    resolution: Any,
    cohort_name: str,
    cs: Any,
) -> dict[str, float | None]:
    """Run per-cohort inference and compute metrics.

    On I/O failure (H5 missing, corrupt checkpoint) or segmentation data
    errors, returns a null-filled metric dict with a ``"status"`` field
    describing the failure.  Programming errors (``TypeError``,
    ``AttributeError``, etc.) are **not** swallowed and propagate to the
    caller — they indicate a code defect, not a runtime data issue.

    Parameters
    ----------
    cfg:
        Frozen :class:`~vena.segmentation.config.SegmentationConfig`.
    best_ckpt:
        Path to the best checkpoint (``best.pt``).
    patient_ids:
        Patient IDs to evaluate for this cohort.
    resolution:
        Split resolution (for ``patient_to_scans`` expansion).
    cohort_name:
        Display name of the cohort (for logging).
    cs:
        :class:`~vena.segmentation.data.fm_splits.CohortSplit` for this cohort.

    Returns
    -------
    dict[str, float | None]
        Keys: ``"status"`` (``"ok"`` or ``"error: <type>: <msg>"``),
        ``"n_evaluated"`` (int), and per-metric float fields (or ``None``
        when inference did not run).
    """
    from vena.segmentation.exceptions import SegDataError  # type: ignore[import]

    if not patient_ids:
        logger.debug("Cohort %s: no eval patients assigned.", cohort_name)
        return _null_metrics(f"error: no patients assigned to cohort '{cohort_name}'")

    try:
        metrics: dict[str, float] = _run_inference_loop(
            cfg=cfg,
            best_ckpt=best_ckpt,
            patient_ids=patient_ids,
            resolution=resolution,
            cohort_name=cohort_name,
            image_h5=cs.image_h5,
        )
        out: dict[str, float | None] = dict(metrics)
        out["status"] = "ok"
        out["n_evaluated"] = len(patient_ids)
        return out
    except (OSError, SegDataError, RuntimeError) as exc:
        status = f"error: {type(exc).__name__}: {exc}"
        logger.warning(
            "Cohort %s: inference failed (%s) — metrics set to null.",
            cohort_name,
            exc,
        )
        return _null_metrics(status)


def _run_inference_loop(
    *,
    cfg: Any,
    best_ckpt: Path,
    patient_ids: tuple[str, ...],
    resolution: Any,
    cohort_name: str,
    image_h5: Path,
) -> dict[str, float]:
    """Core per-cohort inference loop.

    Loads the best checkpoint, iterates over ``patient_ids`` expanded to
    scan IDs, runs sliding-window inference, and returns per-cohort
    aggregate metrics.

    Notes
    -----
    - Uses ``torch.no_grad()`` and clears CUDA cache after each patient.
    - Sliding-window overlap fraction mirrors :attr:`_SWI_OVERLAP` in the trainer.
    - AHD is only computed when ``len(patient_ids) > 0``; raises no errors on
      degenerate cases (empty ground-truth).
    """
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference  # type: ignore[import]

    from vena.segmentation.data.dataset import SegImageDataset  # type: ignore[import]
    from vena.segmentation.engine import load_seg_checkpoint  # type: ignore[import]
    from vena.segmentation.exceptions import SegDataError  # type: ignore[import]
    from vena.segmentation.metrics.calibration import (  # type: ignore[import]
        brier,
        classwise_ece,
    )
    from vena.segmentation.metrics.overlap import (  # type: ignore[import]
        average_hausdorff,
        dice,
        et_diagnostic,
    )

    device = torch.device(cfg.run.device if hasattr(cfg.run, "device") else "cuda")
    dev_type = device.type

    # ---- load model — canonical loader: strict=True, model_meta-aware -------
    model = load_seg_checkpoint(cfg, best_ckpt, device)

    # ---- expand patient → scan IDs (longitudinal cohorts) ---------------
    patient_to_scans: Mapping[str, tuple[str, ...]] = resolution.patient_to_scans
    scan_ids: list[str] = []
    for pid in patient_ids:
        if pid in patient_to_scans:
            scan_ids.extend(patient_to_scans[pid])
        else:
            scan_ids.append(pid)

    if not scan_ids:
        raise SegDataError(
            f"Cohort '{cohort_name}': patient→scan expansion yielded zero scans "
            f"for {len(patient_ids)} patient(s)."
        )

    # ---- build dataset --------------------------------------------------
    ds = SegImageDataset(
        ids=scan_ids,
        cfg=cfg.data,
        augment=False,
        target_cfg=cfg.targets,
    )

    # ---- inference --------------------------------------------------
    patch_size = tuple(cfg.data.patch_size)
    swi_overlap = 0.25

    dice_tc_list: list[float] = []
    dice_netc_list: list[float] = []
    brier_tc_list: list[float] = []
    brier_netc_list: list[float] = []
    ece_tc_list: list[float] = []
    ece_netc_list: list[float] = []
    et_dice_list: list[float] = []
    et_soft_list: list[float] = []
    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []

    use_amp = cfg.train.amp and dev_type == "cuda"
    amp_dtype = torch.float16 if dev_type == "cuda" else torch.bfloat16

    def _predictor(x: Any) -> Any:
        with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
            out = model(x)
            # Deep supervision returns a list; take the full-res output
            return out[0] if isinstance(out, (list, tuple)) else out

    with torch.no_grad():
        for i in range(len(ds)):
            item: dict = ds[i]
            image: Any = item["image"]
            target: Any = item["target"]

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)

            # Add batch dim
            image_b = image.unsqueeze(0).to(device, non_blocking=True)

            # Sliding-window inference
            logits: Any = sliding_window_inference(
                inputs=image_b,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=_predictor,
                overlap=swi_overlap,
                mode="gaussian",
            )
            # Keep as tensors — metric functions (dice, brier, ece, et_diagnostic)
            # all call torch Tensor methods (.float(), .sum(), etc.)
            pred_soft_t = torch.sigmoid(logits).squeeze(0).cpu()  # (2, H, W, D)
            target_t = (
                target.cpu()
                if isinstance(target, torch.Tensor)
                else torch.from_numpy(np.asarray(target))
            ).float()  # (2, H, W, D) soft
            target_hard_t = (target_t >= 0.5).float()  # (2, H, W, D) binarised

            # dice takes per-channel (H,W,D); brier/ece/et_diagnostic take full (2,H,W,D)
            dice_tc = dice(pred_soft_t[0], target_hard_t[0], threshold=0.5)
            dice_netc = dice(pred_soft_t[1], target_hard_t[1], threshold=0.5)
            brier_result = brier(pred_soft_t, target_hard_t)
            brier_tc = brier_result["tc"]
            brier_netc = brier_result["netc"]
            ece_result = classwise_ece(pred_soft_t, target_hard_t)
            ece_tc = ece_result["tc"]
            ece_netc = ece_result["netc"]
            et_diag = et_diagnostic(pred_soft_t, target_t)

            dice_tc_list.append(dice_tc)
            dice_netc_list.append(dice_netc)
            brier_tc_list.append(brier_tc)
            brier_netc_list.append(brier_netc)
            ece_tc_list.append(ece_tc)
            ece_netc_list.append(ece_netc)
            et_dice_list.append(et_diag["et_dice"])
            et_soft_list.append(et_diag["mean_et_soft"])

            # Store tensors for AHD (average_hausdorff expects torch.Tensor)
            pred_list.append(pred_soft_t)
            target_list.append(target_hard_t)

            if dev_type == "cuda":
                torch.cuda.empty_cache()

    if not dice_tc_list:
        raise SegDataError(
            f"Cohort '{cohort_name}': inference loop completed but produced no predictions."
        )

    n = len(dice_tc_list)
    mean_dice_tc = float(sum(dice_tc_list) / n)
    mean_dice_netc = float(sum(dice_netc_list) / n)

    # AHD — average_hausdorff expects torch.Tensor (per-class, any spatial shape)
    try:
        # pred_list / target_list contain full (2, H, W, D) tensors
        ahd_tc_vals = [
            average_hausdorff(p[0], t[0]) for p, t in zip(pred_list, target_list, strict=True)
        ]
        ahd_netc_vals = [
            average_hausdorff(p[1], t[1]) for p, t in zip(pred_list, target_list, strict=True)
        ]
        # NaN means empty mask — exclude from the mean
        ahd_tc_finite = [v for v in ahd_tc_vals if not (v != v)]  # filter nan
        ahd_netc_finite = [v for v in ahd_netc_vals if not (v != v)]
        ahd_tc = float(sum(ahd_tc_finite) / len(ahd_tc_finite)) if ahd_tc_finite else float("nan")
        ahd_netc = (
            float(sum(ahd_netc_finite) / len(ahd_netc_finite)) if ahd_netc_finite else float("nan")
        )
    except (ValueError, RuntimeError) as exc:
        logger.debug("AHD computation failed for cohort '%s': %s", cohort_name, exc)
        ahd_tc = float("nan")
        ahd_netc = float("nan")

    return {
        "tc": mean_dice_tc,
        "netc": mean_dice_netc,
        "tc_ahd": ahd_tc,
        "netc_ahd": ahd_netc,
        "tc_ece": float(sum(ece_tc_list) / n),
        "netc_ece": float(sum(ece_netc_list) / n),
        "tc_brier": float(sum(brier_tc_list) / n),
        "netc_brier": float(sum(brier_netc_list) / n),
        "et_dice": float(sum(et_dice_list) / n),
        "mean_et_soft": float(sum(et_soft_list) / n),
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_resolved_config(run_dir: Path, cfg: Any) -> None:
    """Persist the resolved YAML config next to the run artifacts."""
    config_path = run_dir / "config.resolved.yaml"
    # Pydantic model_dump gives a JSON-serialisable dict
    raw = cfg.model_dump(mode="python")

    # Convert Path objects to strings for safe YAML serialisation
    def _convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        return obj

    config_path.write_text(yaml.dump(_convert(raw), default_flow_style=False), encoding="utf-8")


def _maybe_round(v: float | None, n: int = 6) -> float | None:
    """Round *v* to *n* decimal places; propagate ``None`` unchanged."""
    return round(v, n) if v is not None else None


def _write_decision_json(
    *,
    run_dir: Path,
    cfg: Any,
    run_id: str,
    produced_at: str,
    git_sha: str,
    resolution: Any,
    plan: Any,
    result: Any,
    gseg_result: Any,
    dice_by_cohort: dict[str, dict[str, float]],
) -> Path:
    """Write the machine-readable ``decision.json`` artifact.

    Schema version: 1.0.0 (segmenter — distinct from FM training schema).

    Notes
    -----
    - **No temperature keys** — iter-9 decision Q5 (2026-07-23).
    - ``et_dice`` / ``mean_et_soft`` are REPORTED diagnostics, not gate inputs.
    - Ring-B cohorts appear with ``role: "test_only"``.
    """
    # Compute checkpoint SHA-256
    ckpt_sha = _sha256_file(result.checkpoint) or "unknown"

    # Compute corpus registry SHA-256
    registry_sha = _sha256_file(Path(cfg.data.corpus_registry)) or "unknown"

    # Model checkpoint SHA-256 (initial backbone checkpoint, if any)
    model_ckpt_path = cfg.model.checkpoint
    if model_ckpt_path is not None:
        model_ckpt_sha = _sha256_file(Path(model_ckpt_path))
    else:
        model_ckpt_sha = None

    # Encoder load coverage: attempt to load and compute, else use default
    encoder_coverage = _get_encoder_coverage(cfg)

    # Build per-cohort G-SEG table with role annotation from resolution
    cohort_role: dict[str, str] = {}
    cohort_n: dict[str, int] = {}
    for cs in resolution.per_cohort:
        cohort_role[cs.name] = cs.role
        # n = number of eval patients for this cohort
        if cs.role == "cv":
            fold = cfg.run.fold
            if fold == "all_train":
                eval_ids = frozenset(plan.fm_val_ids) | frozenset(plan.fm_test_ids)
                cohort_n[cs.name] = sum(
                    1 for pid in eval_ids if resolution.patient_to_cohort.get(pid) == cs.name
                )
            else:
                oof_ids = frozenset(plan.folds[fold])
                cohort_n[cs.name] = sum(
                    1 for pid in oof_ids if resolution.patient_to_cohort.get(pid) == cs.name
                )
        else:
            cohort_n[cs.name] = len(cs.test_patients)

    per_cohort_payload: dict[str, dict[str, Any]] = {}
    for cohort_name, metrics in dice_by_cohort.items():
        per_cohort_payload[cohort_name] = {
            "role": cohort_role.get(cohort_name, "unknown"),
            "n": cohort_n.get(cohort_name, 0),
            "status": metrics.get("status", "ok"),
            "n_evaluated": metrics.get("n_evaluated", 0),
            "tc_dice": _maybe_round(metrics.get("tc")),
            "netc_dice": _maybe_round(metrics.get("netc")),
            "tc_ahd": _maybe_round(metrics.get("tc_ahd")),
            "netc_ahd": _maybe_round(metrics.get("netc_ahd")),
            "tc_ece": _maybe_round(metrics.get("tc_ece")),
            "netc_ece": _maybe_round(metrics.get("netc_ece")),
            "tc_brier": _maybe_round(metrics.get("tc_brier")),
            "netc_brier": _maybe_round(metrics.get("netc_brier")),
            # ET diagnostic — reported, not gated
            "et_dice": _maybe_round(metrics.get("et_dice")),
            "mean_et_soft": _maybe_round(metrics.get("mean_et_soft")),
        }

    # Resolved config as JSON string (round-trip provenance)
    raw_cfg = cfg.model_dump(mode="python")

    def _serialise(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(x) for x in obj]
        return obj

    config_json_str = json.dumps(_serialise(raw_cfg))

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "produced_at": produced_at.replace("-", ":").replace("T", "T"),
        "producer": _PRODUCER,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "backbone_arm": cfg.model.name,
        "model_checkpoint": str(model_ckpt_path) if model_ckpt_path is not None else None,
        "model_checkpoint_sha256": model_ckpt_sha,
        "encoder_load_coverage": encoder_coverage,
        "fold": cfg.run.fold,
        "k_folds": cfg.data.k_folds,
        "fold_seed": cfg.data.fold_seed,
        "fm_fold": cfg.data.fm_fold,
        "seed": cfg.run.seed if cfg.run.seed is not None else cfg.seed,
        "corpus_registry": str(cfg.data.corpus_registry),
        "corpus_registry_sha256": registry_sha,
        "dedup_decision_path": (
            str(cfg.data.dedup_decision_path) if cfg.data.dedup_decision_path is not None else None
        ),
        "ckpt_sha256": ckpt_sha,
        "selection_metric": cfg.metrics.selection_metric,
        "best_epoch": result.best_epoch,
        "best_score": float(result.best_score),
        "tumor_region": cfg.targets.tumor_region,
        "gseg": {
            "passed": gseg_result.passed,
            "thresholds": {
                "tc_dice": cfg.metrics.gseg_tc_dice,
                "netc_dice": cfg.metrics.gseg_netc_dice,
            },
            "failures": list(gseg_result.failures),
            "per_cohort": per_cohort_payload,
        },
        "git_sha": git_sha,
        "config_json": config_json_str,
    }

    decision_path = run_dir / "decision.json"
    decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return decision_path


def _get_encoder_coverage(cfg: Any) -> dict[str, Any]:
    """Return encoder load coverage, using known defaults when not computable.

    For the UKB arm: 182/198 = 0.919 (measured and pinned in MEMORY.md).
    For BraTS arm: not pinned; return placeholder.
    For SegResNet: no pretrained encoder; return 0/0.
    """
    known_coverage = {
        "bsf_swinunetr_ukb": {"matched": 182, "total": 198, "fraction": 0.919},
        "bsf_swinunetr_brats": {"matched": 0, "total": 198, "fraction": 0.0},
        "segresnet": {"matched": 0, "total": 0, "fraction": 0.0},
    }
    return known_coverage.get(cfg.model.name, {"matched": 0, "total": 0, "fraction": 0.0})

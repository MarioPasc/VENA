"""Clean per-step and per-epoch training-metrics CSVs.

Replaces Lightning's default ``CSVLogger`` output, which interleaves per-step
train metrics, per-epoch validation metrics, and per-step LR onto disjoint rows
— producing a wide, mostly-empty matrix that also duplicates ``val_epoch.csv``.

Instead we write two tight, fully-populated files:

* ``metrics/train_step.csv`` — one row per *optimiser* step. Columns are
  discovered once (every ``train/*`` key logged by the module + the optimiser
  LR) and then frozen, so every row is fully populated.
* ``metrics/train_epoch.csv`` — one row per training epoch with the mean/std of
  the core losses and throughput across that epoch's optimiser steps, for a
  clean train-vs-val-per-epoch comparison.

Gradient accumulation: ``on_train_batch_end`` fires once per *micro-batch*, but
``trainer.global_step`` advances only on optimiser steps, so we gate the write
on it changing — one row per optimiser step regardless of ``grad_accum``.

Note: ``train/ema_decay`` is logged in the module's ``on_train_batch_end`` which
fires *after* this callback, so its value lags by one optimiser step. It is a
slowly-varying schedule quantity, so the lag is immaterial.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Losses accumulated for the per-epoch summary (column name without ``train/``).
# The schema is the same across stages: keys absent from a given epoch (e.g.
# ``contrastive`` during S1) appear as NaN columns rather than vanishing — so
# train_epoch.csv has a fixed shape across S1 / S2 / S3 runs.
_EPOCH_AGG_KEYS: tuple[str, ...] = (
    "cfm",
    "contrastive",
    "reconstruction",
    "total",
    "samples_per_sec",
    "grad_norm_cn_preclip",
    "grad_norm_cn_postclip",
    "grad_norm_trunk_preclip",
    "grad_norm_trunk_postclip",
    "t_mean",
    "gpu_mem_peak_mb",
)


class TrainMetricsCSV(pl.Callback):
    """Writes ``train_step.csv`` (per optimiser step) and ``train_epoch.csv``."""

    def __init__(self, out_dir: Path | str) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.step_path = self.out_dir / "train_step.csv"
        self.epoch_path = self.out_dir / "train_epoch.csv"
        self._step_header: list[str] | None = None
        # 0 = "no optimiser step written yet"; we write only when global_step
        # advances past it (one row per optimiser step, grad-accum-safe, and
        # skips the pre-first-step accumulation micro-batches at global_step 0).
        self._last_step: int = 0
        self._epoch_accum: dict[str, list[float]] = {}
        # Per-cohort cfm columns are discovered at runtime (P1.2): the order is
        # frozen on first epoch-write so the CSV schema stays stable across
        # subsequent epochs even if one cohort is absent from a particular epoch.
        self._epoch_header: list[str] | None = None
        # Wall-clock start of the current epoch, set in ``on_train_epoch_start``
        # so the per-epoch INFO summary line carries an elapsed-seconds field.
        # That makes the SLURM ``.out`` tail show convergence pace at a glance.
        self._epoch_t0: float | None = None

    # ------------------------------------------------------------------
    # Per-epoch start (timer)
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Per-step
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args: object
    ) -> None:
        step = int(trainer.global_step)
        if step <= self._last_step:
            return  # no optimiser step happened (gradient accumulation / pre-step)
        self._last_step = step

        scalars = self._collect_train_scalars(trainer)
        if not scalars:
            return
        scalars["lr"] = self._current_lr(trainer)
        # ``train/ema_decay`` is logged in the module's ``on_train_batch_end``,
        # which fires *after* this callback, so it is absent from
        # ``callback_metrics`` on the first written row (and would be dropped
        # from the frozen header). Read it live from the module instead so the
        # column always exists.
        ema = getattr(pl_module, "ema", None)
        if ema is not None:
            try:
                scalars["ema_decay"] = float(ema.get_current_decay())
            except (AttributeError, RuntimeError) as exc:
                # ema-pytorch raises RuntimeError before any update_after_step
                # has been reached; AttributeError covers a hypothetical EMA
                # implementation without the helper. Log so an unexpected
                # failure is visible instead of silently dropping the column.
                logger.debug("ema_decay unavailable: %s", exc)

        if self._step_header is None:
            self._step_header = ["epoch", "step", "lr", *sorted(scalars.keys() - {"lr"})]
            if not self.step_path.exists():
                with self.step_path.open("w", newline="") as f:
                    csv.writer(f).writerow(self._step_header)

        row = [int(trainer.current_epoch), step]
        row += [_f(scalars.get(c)) for c in self._step_header[2:]]
        with self.step_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

        for k in _EPOCH_AGG_KEYS:
            if k in scalars:
                self._epoch_accum.setdefault(k, []).append(scalars[k])
        # P1.2 — also accumulate any per-cohort cfm key discovered at runtime.
        # The module logs ``train/cfm_cohort_<sanitised-name>`` per step when
        # the batch carries cohort tags; these keys are absent for
        # single-cohort runs.
        for k in scalars:
            if k.startswith("cfm_cohort_"):
                self._epoch_accum.setdefault(k, []).append(scalars[k])

    # ------------------------------------------------------------------
    # Per-epoch
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._epoch_accum:
            return
        epoch = int(trainer.current_epoch)
        # Freeze the per-cohort key order the first time we write — so the
        # epoch CSV has a stable schema even if a cohort is missing from a
        # later epoch's batches (unusual but possible with extreme sampling).
        if self._epoch_header is None:
            cohort_keys = sorted(k for k in self._epoch_accum if k.startswith("cfm_cohort_"))
            self._epoch_header = list(_EPOCH_AGG_KEYS) + cohort_keys
        cols = ["epoch", "step", "n_steps"]
        for k in self._epoch_header:
            cols += [f"{k}_mean", f"{k}_std"]
        if not self.epoch_path.exists():
            with self.epoch_path.open("w", newline="") as f:
                csv.writer(f).writerow(cols)
        n_steps = max((len(v) for v in self._epoch_accum.values()), default=0)
        row: list[object] = [epoch, int(trainer.global_step), n_steps]
        for k in self._epoch_header:
            xs = self._epoch_accum.get(k, [])
            m, s = _mean_std(xs)
            row += [_f(m), _f(s)]
        with self.epoch_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

        # Per-epoch human-readable summary into both ``train.log`` and the
        # SLURM ``.out``. Aggregates already in self._epoch_accum; pull the
        # mean of the load-bearing keys, fall back to NaN-tolerant ``-``.
        def _m(k: str) -> float:
            m, _ = _mean_std(self._epoch_accum.get(k, []))
            return float("nan") if m is None else float(m)

        elapsed = (
            time.perf_counter() - self._epoch_t0 if self._epoch_t0 is not None else float("nan")
        )
        logger.info(
            "epoch %d done | step=%d total=%.4f cfm=%.4f contrastive=%.4f "
            "gpu_peak_mb=%.0f samples_per_sec=%.2f elapsed=%.1fs",
            epoch,
            int(trainer.global_step),
            _m("total"),
            _m("cfm"),
            _m("contrastive"),
            _m("gpu_mem_peak_mb"),
            _m("samples_per_sec"),
            elapsed,
        )

        self._epoch_accum.clear()
        self._epoch_t0 = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_train_scalars(trainer: pl.Trainer) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, val in trainer.callback_metrics.items():
            if not key.startswith("train/"):
                continue
            try:
                out[key[len("train/") :]] = float(val)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _current_lr(trainer: pl.Trainer) -> float | None:
        if not trainer.optimizers:
            return None
        return float(trainer.optimizers[0].param_groups[0]["lr"])


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    finite = [x for x in xs if x is not None and not math.isnan(x)]
    if not finite:
        return None, None
    m = sum(finite) / len(finite)
    if len(finite) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in finite) / (len(finite) - 1))


def _f(v: float | None) -> str:
    if v is None:
        return ""
    try:
        if v != v:  # NaN
            return ""
    except TypeError:
        return ""
    return f"{float(v):.6g}"

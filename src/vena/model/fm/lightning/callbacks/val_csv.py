"""Append-only writer for ``val_epoch.csv`` in long format.

Schema per training_routine.md §4.4. The LightningModule populates a
per-region accumulator in ``validation_step`` (a dict keyed by ``(nfe,
region)`` holding raw per-patient lists). This callback collapses it to
mean/std via ``pl_module.collapse_val_metrics()`` on
``on_validation_epoch_end`` and flushes one row per ``(nfe, region)`` to the
CSV. The accumulator is cleared by the module's own (later-firing)
``on_validation_epoch_end`` hook, not here.

Resume semantics: on ``on_train_start``, if a CSV exists, truncate any rows
with ``epoch > resumed_epoch`` to keep the file monotonic and free of
duplicates.
"""

from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


COLUMNS: tuple[str, ...] = (
    "epoch",
    "step",
    "nfe",
    "region",
    "mse_latent_mean",
    "mse_latent_std",
    "l1_latent_mean",
    "l1_latent_std",
    "cosine_latent_mean",
    "psnr_image_mean",
    "psnr_image_std",
    "ssim_image_mean",
    "ssim_image_std",
    "n_patients",
    "n_image_patients",
    "timestamp_utc",
)


class ValMetricsCSV(pl.Callback):
    """Writes ``metrics/val_epoch.csv`` in append mode."""

    def __init__(self, csv_path: Path | str) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Truncate rows past the resumed epoch (idempotent on fresh runs)."""
        resumed_epoch = int(trainer.current_epoch)
        self._truncate_past_epoch(resumed_epoch)

    def _truncate_past_epoch(self, epoch_inclusive: int) -> None:
        if not self.csv_path.exists():
            return
        kept: list[list[str]] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                try:
                    if int(row[0]) <= epoch_inclusive:
                        kept.append(row)
                except (ValueError, IndexError):
                    continue
        if header is None:
            return
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(kept)
        logger.info(
            "ValMetricsCSV: truncated to epoch ≤ %d (%d rows kept)",
            epoch_inclusive,
            len(kept),
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # The module collapses the raw per-region accumulator (lists of
        # per-patient values) into mean/std stats on demand. We must NOT read
        # ``_val_accumulator`` directly: Lightning fires this callback hook
        # *before* ``LightningModule.on_validation_epoch_end``, so at this point
        # the accumulator still holds raw lists, not the ``*_mean``/``*_std``
        # keys this CSV needs. The module clears the accumulator in its own
        # (later) hook, so we do not clear here.
        collapse = getattr(pl_module, "collapse_val_metrics", None)
        if collapse is None:
            return
        metrics = collapse()
        if not metrics:
            return
        epoch = int(trainer.current_epoch)
        step = int(trainer.global_step)
        ts = datetime.now(UTC).isoformat()
        rows: list[list[str]] = []
        for (nfe, region), agg in metrics.items():
            rows.append(
                [
                    epoch,
                    step,
                    nfe,
                    region,
                    _f(agg.get("mse_latent_mean")),
                    _f(agg.get("mse_latent_std")),
                    _f(agg.get("l1_latent_mean")),
                    _f(agg.get("l1_latent_std")),
                    _f(agg.get("cosine_latent_mean")),
                    _f(agg.get("psnr_image_mean")),
                    _f(agg.get("psnr_image_std")),
                    _f(agg.get("ssim_image_mean")),
                    _f(agg.get("ssim_image_std")),
                    int(agg.get("n_patients", 0)),
                    int(agg.get("n_image_patients", 0)),
                    ts,
                ]
            )
        if not rows:
            return
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerows(rows)


def _f(v: float | None) -> str:
    if v is None:
        return ""
    try:
        if v != v:  # NaN
            return ""
    except TypeError:
        pass
    return f"{float(v):.6g}"

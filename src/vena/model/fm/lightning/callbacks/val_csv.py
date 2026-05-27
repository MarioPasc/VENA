"""Append-only writer for ``val_epoch.csv`` in long format.

Schema per training_routine.md §4.4. The LightningModule populates a
per-region accumulator in ``validation_step`` and exposes it as
``trainer.lightning_module._val_accumulator`` (a dict keyed by ``(epoch, nfe,
region)``). This callback consumes the accumulator on
``on_validation_epoch_end`` and flushes one row per tuple to the CSV.

Resume semantics: on ``on_train_start``, if a CSV exists, truncate any rows
with ``epoch > resumed_epoch`` to keep the file monotonic and free of
duplicates.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


COLUMNS: tuple[str, ...] = (
    "epoch", "step", "nfe", "region",
    "mse_latent_mean", "mse_latent_std",
    "l1_latent_mean", "l1_latent_std",
    "cosine_latent_mean",
    "psnr_image_mean", "psnr_image_std",
    "ssim_image_mean", "ssim_image_std",
    "n_patients",
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

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        accumulator = getattr(pl_module, "_val_accumulator", None)
        if accumulator is None:
            return
        epoch = int(trainer.current_epoch)
        step = int(trainer.global_step)
        ts = datetime.now(timezone.utc).isoformat()
        rows: list[list[str]] = []
        for (nfe, region), agg in accumulator.items():
            rows.append([
                epoch, step, nfe, region,
                _f(agg.get("mse_latent_mean")), _f(agg.get("mse_latent_std")),
                _f(agg.get("l1_latent_mean")), _f(agg.get("l1_latent_std")),
                _f(agg.get("cosine_latent_mean")),
                _f(agg.get("psnr_image_mean")), _f(agg.get("psnr_image_std")),
                _f(agg.get("ssim_image_mean")), _f(agg.get("ssim_image_std")),
                int(agg.get("n_patients", 0)),
                ts,
            ])
        if not rows:
            return
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        # Clear after flush so subsequent epochs start fresh.
        accumulator.clear()


def _f(v: float | None) -> str:
    if v is None:
        return ""
    try:
        if v != v:  # NaN
            return ""
    except TypeError:
        pass
    return f"{float(v):.6g}"

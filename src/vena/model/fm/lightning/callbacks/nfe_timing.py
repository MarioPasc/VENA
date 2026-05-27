"""Per-epoch NFE timing CSV (training_routine.md §5).

Reads from ``pl_module._nfe_timing_buffer``, a list of dicts populated by the
LightningModule's validation step during sweep epochs. One row per
``(epoch, nfe)``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = (
    "epoch", "nfe",
    "t_trunk_mean_sec", "t_controlnet_mean_sec",
    "t_decode_sec",
    "t_total_mean_sec", "t_total_std_sec",
    "gpu_mem_peak_mb",
    "n_patients_measured",
)


class NFETimingCSV(pl.Callback):
    """Writes ``performance/nfe_timing_epoch_{NNN}.csv`` after a sweep epoch."""

    def __init__(self, out_dir: Path | str) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        buf: list[dict] | None = getattr(pl_module, "_nfe_timing_buffer", None)
        if not buf:
            return
        epoch = int(trainer.current_epoch)
        path = self.out_dir / f"nfe_timing_epoch_{epoch:03d}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            for row in buf:
                writer.writerow([row.get(c, "") for c in COLUMNS])
        logger.info("NFETimingCSV: wrote %d rows to %s", len(buf), path)
        buf.clear()

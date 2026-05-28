"""Per-epoch NFE timing CSV (training_routine.md §5).

The LightningModule accumulates per-component timings (trunk / controlnet /
decode) across validation batches into ``_nfe_timing_accum`` and exposes the
aggregated rows via ``pl_module.collapse_nfe_timing()``. This callback writes
one row per ``(epoch, nfe)`` on ``on_validation_epoch_end``. The accumulator is
cleared by the module's own (later-firing) ``on_validation_epoch_end`` hook,
not here — see the ordering note in ``val_csv.py``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = (
    "epoch",
    "nfe",
    "t_trunk_mean_sec",
    "t_controlnet_mean_sec",
    "t_decode_sec",
    "t_total_mean_sec",
    "t_total_std_sec",
    "gpu_mem_peak_mb",
    "n_patients_measured",
)


class NFETimingCSV(pl.Callback):
    """Writes ``performance/nfe_timing_epoch_{NNN}.csv`` after a sweep epoch."""

    def __init__(self, out_dir: Path | str) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        collapse = getattr(pl_module, "collapse_nfe_timing", None)
        if collapse is None:
            return
        rows = collapse()
        if not rows:
            return
        epoch = int(trainer.current_epoch)
        path = self.out_dir / f"nfe_timing_epoch_{epoch:03d}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            for row in rows:
                row = {**row, "epoch": epoch}
                writer.writerow([_fmt(row.get(c)) for c in COLUMNS])
        logger.info("NFETimingCSV: wrote %d rows to %s", len(rows), path)


def _fmt(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return f"{v:.6g}"
    return str(v)

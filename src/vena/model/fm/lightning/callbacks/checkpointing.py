"""Checkpoint callback — wraps :class:`pytorch_lightning.callbacks.ModelCheckpoint`.

Lightning's built-in :class:`ModelCheckpoint` already provides:

* Atomic writes via tmp-then-rename.
* Best/top-k retention (``save_top_k``, ``monitor``, ``mode``).
* "Last" pointer (``save_last=True``).

This subclass adds:

* Default filename template ``ema_epoch_{epoch:03d}`` (Lightning appends
  ``.ckpt``; we accept the suffix departure from the doc's ``.pt``).
* A ``best_metric_name`` / ``best_metric_region`` resolver so the YAML can
  declare e.g. ``mse_latent`` + ``bg`` and the callback monitors
  ``val/mse_latent_bg_nfe5`` — the canonical key the LightningModule logs.

The module also hosts :class:`TrunkEMASnapshotCallback` — a sibling callback
that mirrors the trunk-EMA shadow next to the Lightning checkpoint on every
save, so the S1→S3 warm-start path (``model-coding-standards.md`` R6 +
``training-stages.md`` §4.5) can restore the fine-tuned trunk EMA byte-for-byte.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

TRUNK_EMA_SNAPSHOT_FILENAME: str = "trunk_ema_snapshot.pt"


class VENACheckpointCallback(ModelCheckpoint):
    """Checkpoint policy for VENA training runs."""

    def __init__(
        self,
        dirpath: Path | str,
        *,
        retention_n_checkpoints: int = 3,
        every_n_epochs: int = 5,
        best_metric_name: str = "mse_latent",
        best_metric_region: str = "bg",
        best_metric_nfe: int = 5,
        best_mode: str = "min",
        filename: str = "ema_epoch_{epoch:03d}",
        monitor_key: str | None = None,
        save_on_train_epoch_end: bool = False,
    ) -> None:
        # ``monitor_key`` overrides the val-metric key — used when in-process
        # validation is offloaded and selection is on a train-epoch metric.
        monitor = monitor_key or (
            f"val/{best_metric_name}_{best_metric_region}_nfe{best_metric_nfe}"
        )
        super().__init__(
            dirpath=str(dirpath),
            filename=filename,
            monitor=monitor,
            mode=best_mode,
            save_top_k=int(retention_n_checkpoints),
            save_last=True,
            every_n_epochs=int(every_n_epochs),
            auto_insert_metric_name=False,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        self.best_metric_key = monitor


class BestCheckpointCallback(ModelCheckpoint):
    """Tracks the single best checkpoint at a stable path ``ema_best.ckpt``.

    Complements :class:`VENACheckpointCallback` (which keeps the rotated
    top-k ``ema_epoch_{NNN}`` files and ``last.ckpt``). This callback writes
    only the current best epoch to a fixed filename, evaluated every epoch so
    the "best" pointer is always current regardless of the retention cadence.
    The resume logic (`resume_from: best`) reads this exact path.
    """

    def __init__(
        self,
        dirpath: Path | str,
        *,
        best_metric_name: str = "mse_latent",
        best_metric_region: str = "bg",
        best_metric_nfe: int = 5,
        best_mode: str = "min",
        monitor_key: str | None = None,
        save_on_train_epoch_end: bool = False,
    ) -> None:
        monitor = monitor_key or (
            f"val/{best_metric_name}_{best_metric_region}_nfe{best_metric_nfe}"
        )
        super().__init__(
            dirpath=str(dirpath),
            filename="ema_best",
            monitor=monitor,
            mode=best_mode,
            save_top_k=1,
            save_last=False,
            every_n_epochs=1,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        self.best_metric_key = monitor


class TrunkEMASnapshotCallback(pl.Callback):
    """Mirror the trunk-EMA shadow to ``<dirpath>/trunk_ema_snapshot.pt`` on save.

    Bridges a gap in the warm-start path: Lightning's checkpoint payload carries
    the trunk-EMA shadow as a registered submodule, but a WARM_START load only
    transfers the live trunk + ControlNet weights and re-initialises the EMA
    shadow from scratch in :meth:`FMLightningModule.setup`. Without a sibling
    snapshot, the S1→S3 warm-start would silently restart the trunk-EMA
    averaging, breaking the warm-start-vs-scratch ablation
    (model-coding-standards.md R6, training-stages.md §4.5).

    The callback fires once per Lightning checkpoint save event. The snapshot
    is a single shared file per run directory (overwritten each save) — match
    the format written by ``ExhaustiveValLauncher._launch`` so the existing
    snapshot consumer in :mod:`routines.fm.exhaustive_val.engine` keeps
    working unchanged.

    The callback is a no-op when ``pl_module.trunk_ema is None`` (frozen-trunk
    runs), so it is safe to attach unconditionally.
    """

    def __init__(
        self,
        dirpath: Path | str,
        *,
        filename: str = TRUNK_EMA_SNAPSHOT_FILENAME,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        trunk_ema = getattr(pl_module, "trunk_ema", None)
        if trunk_ema is None:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        target = self.dirpath / self.filename
        # Atomic-ish replace via tmp file; cheap on a single shadow state_dict.
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(trunk_ema.ema_model.state_dict(), tmp)
        tmp.replace(target)
        logger.debug("TrunkEMASnapshotCallback wrote %s", target)

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
"""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


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

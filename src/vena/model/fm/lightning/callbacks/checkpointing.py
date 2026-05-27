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
    ) -> None:
        monitor_key = f"val/{best_metric_name}_{best_metric_region}_nfe{best_metric_nfe}"
        super().__init__(
            dirpath=str(dirpath),
            filename=filename,
            monitor=monitor_key,
            mode=best_mode,
            save_top_k=int(retention_n_checkpoints),
            save_last=True,
            every_n_epochs=int(every_n_epochs),
            auto_insert_metric_name=False,
            # The monitored key is populated by ``on_validation_epoch_end``;
            # firing on train-epoch-end runs before val and misses the key.
            save_on_train_epoch_end=False,
        )
        self.best_metric_key = monitor_key

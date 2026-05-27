"""Signal handler that writes ``ema_final.pt`` on SIGTERM.

HPC schedulers send SIGTERM at wall-clock expiry. Lightning will not save a
checkpoint by default — we register a handler that calls
``trainer.save_checkpoint(...)`` and exits 0 so the next launch can resume
via ``run.resume_from: latest``.

The callback registers/deregisters the handler on
:meth:`on_train_start` / :meth:`on_train_end` so it does not leak across
multiple trainer.fit() invocations.
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class SigtermHandler(pl.Callback):
    """SIGTERM → save ``ema_final.pt`` and exit 0."""

    def __init__(self, ckpt_dir: Path | str, filename: str = "ema_final.ckpt") -> None:
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self.filename = filename
        self._prev_handler = None
        self._trainer: pl.Trainer | None = None

    def _handler(self, signum, frame) -> None:  # noqa: ANN001 — signal signature
        if self._trainer is None:
            sys.exit(1)
        path = self.ckpt_dir / self.filename
        logger.warning("SIGTERM received — saving %s and exiting.", path)
        try:
            self._trainer.save_checkpoint(str(path))
        except Exception as exc:  # noqa: BLE001 — last-ditch save
            logger.error("save_checkpoint failed on SIGTERM: %s", exc)
        sys.exit(0)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._trainer = trainer
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._prev_handler = signal.signal(signal.SIGTERM, self._handler)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._prev_handler is not None:
            signal.signal(signal.SIGTERM, self._prev_handler)
        self._trainer = None

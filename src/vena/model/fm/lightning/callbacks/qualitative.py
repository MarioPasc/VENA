"""Qualitative-latent dump per training_routine.md §6.

The LightningModule places `(patient_id, nfe) -> latent_fp16` entries into a
dict ``pl_module._qualitative_buffer`` during ``validation_step`` when the
epoch matches ``qualitative_every_epochs``. This callback flushes the buffer
into ``qualitative/epoch_{NNN:03d}.h5`` at ``on_validation_epoch_end`` and
clears it.

File format:

* Root attrs: ``epoch``, ``step``, ``run_id``, ``timestamp_utc``.
* Group per patient: ``/predictions/{patient_id}``.
* Dataset per NFE under that group: ``nfe_{N}`` of shape ``(C, h, w, d)``
  fp16 gzip-4.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)


class QualitativeH5Writer(pl.Callback):
    """Writes ``qualitative/epoch_{NNN}.h5`` when the buffer is populated."""

    def __init__(self, out_dir: Path | str, run_id: str) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        buf: dict[tuple[str, int], torch.Tensor] | None = getattr(
            pl_module, "_qualitative_buffer", None
        )
        if not buf:
            return
        epoch = int(trainer.current_epoch)
        path = self.out_dir / f"epoch_{epoch:03d}.h5"
        with h5py.File(path, "a") as f:
            if "predictions" not in f:
                f.attrs["epoch"] = epoch
                f.attrs["step"] = int(trainer.global_step)
                f.attrs["run_id"] = self.run_id
                f.attrs["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                f.create_group("predictions")
            grp_root = f["predictions"]
            for (pid, nfe), z in buf.items():
                grp = grp_root.require_group(pid)
                grp.attrs["patient_id"] = pid
                key = f"nfe_{int(nfe)}"
                if key in grp:
                    del grp[key]
                grp.create_dataset(
                    key,
                    data=np.ascontiguousarray(z.cpu().numpy().astype(np.float16)),
                    dtype="float16",
                    compression="gzip",
                    compression_opts=4,
                )
        logger.info(
            "QualitativeH5Writer: wrote %d (patient,nfe) entries to %s",
            len(buf),
            path,
        )
        buf.clear()

"""Launches the exhaustive image-space validation as an async second-GPU job.

On a slow cadence (``every_epochs``) this callback, at the end of a training
epoch, snapshots the current EMA shadow weights and spawns the standalone
``routines.fm.exhaustive_val`` CLI on ``device`` (default ``cuda:1``). Training
continues uninterrupted on the primary GPU while the subprocess samples,
decodes, scores against the real T1c, and renders figures.

Concurrency policy: at most one exhaustive validation in flight. If the
previous one is still running when the cadence fires again, this trigger is
skipped (logged) rather than blocking training or piling onto the second GPU.
In real runs the cadence (~every 20 epochs) is far wider than one validation,
so skips do not occur; they only happen under the every-epoch stress test.
``on_fit_end`` joins the last subprocess so its artifacts are complete.

Whenever a validation launches, the per-device GPU memory of *every* visible
device is appended to ``exhaustive_val/gpu_usage.log`` — evidence that the
training GPU and the validation GPU are busy simultaneously.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml

logger = logging.getLogger(__name__)


class ExhaustiveValLauncher(pl.Callback):
    """Spawns the async exhaustive-validation subprocess on a cadence."""

    def __init__(
        self,
        *,
        run_dir: Path | str,
        run_id: str,
        job_base: dict[str, Any],
        every_epochs: int,
        device: str,
        cwd: Path | str,
        python_executable: str | None = None,
    ) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.job_base = dict(job_base)
        self.every_epochs = int(every_epochs)
        self.device = device
        self.cwd = Path(cwd)
        self.python = python_executable or sys.executable
        self.out_root = self.run_dir / "exhaustive_val"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.gpu_log = self.out_root / "gpu_usage.log"
        self._proc: subprocess.Popen | None = None
        self._proc_epoch: int | None = None

    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.every_epochs <= 0:
            return
        epoch = int(trainer.current_epoch)
        if epoch % self.every_epochs != 0:
            return
        if self._proc is not None and self._proc.poll() is None:
            logger.warning(
                "ExhaustiveValLauncher: previous validation (epoch %s) still running; "
                "skipping epoch %d trigger.",
                self._proc_epoch,
                epoch,
            )
            return
        self._launch(trainer, pl_module, epoch)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._proc is not None and self._proc.poll() is None:
            logger.info(
                "ExhaustiveValLauncher: waiting for exhaustive validation (epoch %s) to finish...",
                self._proc_epoch,
            )
            self._proc.wait()
        if self._proc is not None:
            logger.info(
                "ExhaustiveValLauncher: last validation (epoch %s) exit code %s.",
                self._proc_epoch,
                self._proc.returncode,
            )

    # ------------------------------------------------------------------

    def _launch(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch: int) -> None:
        epoch_dir = self.out_root / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the EMA shadow weights (small; no optimiser state).
        snapshot = epoch_dir / "ema_snapshot.pt"
        torch.save(pl_module.ema.ema_model.state_dict(), snapshot)

        job = dict(self.job_base)
        job.update(
            {
                "run_id": self.run_id,
                "epoch": int(epoch),
                "ema_snapshot": str(snapshot),
                "output_dir": str(epoch_dir),
                "device": self.device,
            }
        )
        job_yaml = epoch_dir / "job.yaml"
        with job_yaml.open("w") as f:
            yaml.safe_dump(job, f, sort_keys=False)

        self._log_gpu_usage(epoch)

        sublog = epoch_dir / "subprocess.log"
        cmd = [self.python, "-m", "routines.fm.exhaustive_val.cli", str(job_yaml)]
        log_fh = sublog.open("w")
        self._proc = subprocess.Popen(
            cmd, cwd=str(self.cwd), stdout=log_fh, stderr=subprocess.STDOUT
        )
        self._proc_epoch = epoch
        logger.info(
            "ExhaustiveValLauncher: launched exhaustive validation epoch=%d pid=%d on %s (log: %s)",
            epoch,
            self._proc.pid,
            self.device,
            sublog,
        )

    def _log_gpu_usage(self, epoch: int) -> None:
        ts = datetime.now(UTC).isoformat()
        lines = [f"[{ts}] epoch={epoch} validation launching on {self.device}"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used_mb = (total - free) / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                name = torch.cuda.get_device_name(i)
                lines.append(f"    cuda:{i} ({name}) used={used_mb:.0f}MB / {total_mb:.0f}MB")
        with self.gpu_log.open("a") as f:
            f.write("\n".join(lines) + "\n")

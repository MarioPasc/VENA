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
        block_until_complete: bool = False,
        prune_snapshots_keep: int = 0,
    ) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.job_base = dict(job_base)
        self.every_epochs = int(every_epochs)
        self.device = device
        # When True, join each launched validation before training continues, so
        # every cadence epoch gets a completed exhaustive pass (used for short
        # diagnostic runs where epochs are far faster than one validation, which
        # would otherwise trip the skip-if-busy guard). Default False keeps the
        # production async, non-blocking behaviour.
        self.block_until_complete = bool(block_until_complete)
        # Number of most-recent exhaustive epoch dirs whose EMA snapshots are
        # kept on disk. 0 disables pruning (keep everything — useful for short
        # diagnostic runs). The ``metrics.csv``, ``timing.csv`` and
        # ``latent_preds.h5`` files are NEVER pruned; only the ~1 GB
        # ``ema_snapshot.pt`` / ``trunk_ema_snapshot.pt`` files are.
        self.prune_snapshots_keep = int(prune_snapshots_keep)
        self.cwd = Path(cwd)
        self.python = python_executable or sys.executable
        self.out_root = self.run_dir / "exhaustive_val"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.gpu_log = self.out_root / "gpu_usage.log"
        self._proc: subprocess.Popen | None = None
        self._proc_epoch: int | None = None
        # Epochs whose metrics.csv has already been summarised into the parent
        # ``train.log``; ensures the summary is logged exactly once per epoch
        # even if both the blocking branch and the next-epoch poll see the
        # completed process.
        self._summarised_epochs: set[int] = set()

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
        # Async mode: if the previous proc finished between cadences, surface
        # its PSNR/SSIM into the parent log before launching the next one.
        self._summarise_if_ready()
        self._launch(trainer, pl_module, epoch)
        if self.block_until_complete and self._proc is not None:
            logger.info(
                "ExhaustiveValLauncher: blocking until epoch %d validation completes...", epoch
            )
            rc = self._proc.wait()
            logger.info("ExhaustiveValLauncher: epoch %d validation exit code %s.", epoch, rc)
            self._summarise_if_ready()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._proc is not None and self._proc.poll() is None:
            logger.info(
                "ExhaustiveValLauncher: waiting for exhaustive validation (epoch %s) to finish...",
                self._proc_epoch,
            )
            self._proc.wait()
        self._summarise_if_ready()
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

        # Unfrozen-trunk ablation: also snapshot the trunk EMA shadow so the
        # exhaustive job samples with the fine-tuned trunk rather than the
        # original frozen checkpoint. Absent when the trunk is frozen.
        trunk_ema = getattr(pl_module, "trunk_ema", None)
        if trunk_ema is not None:
            trunk_snapshot = epoch_dir / "trunk_ema_snapshot.pt"
            torch.save(trunk_ema.ema_model.state_dict(), trunk_snapshot)
            job["trunk_finetuned_snapshot"] = str(trunk_snapshot)
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
        # Prune stale snapshots from older epoch dirs. We trigger at *launch*
        # time so the just-launched subprocess is free to read its own snapshot;
        # by construction the previous validation has finished by now (we
        # checked ``self._proc.poll()`` before launching).
        self._prune_old_snapshots(current_epoch=epoch)

    def _prune_old_snapshots(self, current_epoch: int) -> None:
        """Delete EMA snapshots from epoch dirs older than the keep window.

        Keeps the ``prune_snapshots_keep`` most-recent epoch dirs (sorted by
        epoch number, including the just-launched one). All older epoch dirs
        keep their CSV/H5/figure artifacts but lose their snapshots.
        """
        if self.prune_snapshots_keep <= 0:
            return

        # Sort by numeric epoch, not by directory name. ``_launch`` writes the
        # dir as f"epoch_{epoch:03d}" (min-width 3), so ``epoch_975`` and
        # ``epoch_1000`` are both valid but lex-sort places ``epoch_1000``
        # between ``epoch_100`` and ``epoch_125``. With ``prune_snapshots_keep=2``
        # the freshly-launched ≥1000 epoch dir then falls outside the keep
        # window and its ``ema_snapshot.pt`` is deleted before the spawned
        # subprocess can ``torch.load`` it (FileNotFoundError on long runs).
        def _epoch_key(p: Path) -> int:
            try:
                return int(p.name.removeprefix("epoch_"))
            except ValueError:
                return -1

        epoch_dirs = sorted(
            (d for d in self.out_root.glob("epoch_*") if d.is_dir()),
            key=_epoch_key,
        )
        if len(epoch_dirs) <= self.prune_snapshots_keep:
            return
        to_prune = epoch_dirs[: -self.prune_snapshots_keep]
        freed_mb = 0.0
        for d in to_prune:
            for name in ("ema_snapshot.pt", "trunk_ema_snapshot.pt"):
                p = d / name
                if p.exists():
                    freed_mb += p.stat().st_size / (1024 * 1024)
                    try:
                        p.unlink()
                    except OSError as exc:
                        logger.warning("ExhaustiveValLauncher: could not prune %s (%s)", p, exc)
        if freed_mb > 0:
            logger.info(
                "ExhaustiveValLauncher: pruned %d stale snapshot dir(s), freed %.1f MB.",
                len(to_prune),
                freed_mb,
            )

    def _summarise_if_ready(self) -> None:
        """Read the most-recent completed validation's ``metrics.csv`` and
        emit one human-readable INFO line per NFE (mean ± std PSNR / SSIM /
        latent_mse / gen_sec).

        Idempotent: keyed on ``self._proc_epoch`` so the summary fires
        exactly once even when both the next-epoch poll and ``on_fit_end``
        see the completed process.

        Failures (missing file, malformed CSV) are logged at WARNING and
        swallowed — the summary is a convenience for the SLURM ``.out``
        tail and must never bring down a 7-day training run.
        """
        if self._proc is None or self._proc_epoch is None:
            return
        if self._proc.poll() is None:
            return  # still running
        epoch = self._proc_epoch
        if epoch in self._summarised_epochs:
            return
        rc = self._proc.returncode
        if rc != 0:
            logger.warning(
                "ExhaustiveValLauncher: epoch %d validation exited non-zero (rc=%s); "
                "no summary written.",
                epoch,
                rc,
            )
            self._summarised_epochs.add(epoch)
            return

        metrics_csv = self.out_root / f"epoch_{epoch:03d}" / "metrics.csv"
        if not metrics_csv.exists():
            logger.warning(
                "ExhaustiveValLauncher: epoch %d completed but %s is missing; no summary.",
                epoch,
                metrics_csv,
            )
            self._summarised_epochs.add(epoch)
            return

        try:
            import pandas as pd

            df = pd.read_csv(metrics_csv)
        except (OSError, ValueError) as exc:
            logger.warning(
                "ExhaustiveValLauncher: epoch %d metrics.csv unreadable (%s); no summary.",
                epoch,
                exc,
            )
            self._summarised_epochs.add(epoch)
            return

        required = {"nfe", "psnr_db", "ssim", "latent_mse", "gen_sec"}
        if not required.issubset(df.columns):
            logger.warning(
                "ExhaustiveValLauncher: epoch %d metrics.csv missing columns %s; no summary.",
                epoch,
                sorted(required - set(df.columns)),
            )
            self._summarised_epochs.add(epoch)
            return

        logger.info(
            "exhaustive-val epoch %d summary (n_rows=%d, source=%s):",
            epoch,
            len(df),
            metrics_csv,
        )
        for nfe in sorted(df["nfe"].unique().tolist()):
            sub = df[df["nfe"] == nfe]
            logger.info(
                "  NFE=%d  PSNR=%.2f±%.2f dB  SSIM=%.3f±%.3f  "
                "latent_mse=%.4g  gen_sec=%.2f  (n=%d)",
                int(nfe),
                float(sub["psnr_db"].mean()),
                float(sub["psnr_db"].std()),
                float(sub["ssim"].mean()),
                float(sub["ssim"].std()),
                float(sub["latent_mse"].mean()),
                float(sub["gen_sec"].mean()),
                len(sub),
            )
        self._summarised_epochs.add(epoch)

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

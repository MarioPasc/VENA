"""Flow-matching training routine engine (v2).

Implements the full training_routine.md spec:

* Run directory layout (`experiments/{run_id}/` per §2.2) — atomic, self-contained.
* Validation cadence (per-epoch single-NFE + sweep every K epochs).
* Qualitative-latent dumps, NFE timing, per-region metrics, EMA, RNG-state-in-checkpoint
  resume, SIGTERM-aware checkpointing.
* YAML schema mirroring §2.3 with a top-level ``regions:`` block declaring per-region
  source so the audit trail makes the measurement scope explicit.

The engine is a thin orchestrator: it wires Lightning's Trainer to the
:class:`vena.model.fm.lightning.module.FMLightningModule`, the
:class:`vena.model.fm.lightning.data.LatentH5DataModule`, the VAE decoder, and
the suite of custom callbacks in :mod:`vena.model.fm.lightning.callbacks`.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vena.data.h5.shared import now_iso_utc
from vena.model.fm.lightning import FMLightningModule, LatentH5DataModule
from vena.model.fm.lightning.callbacks import (
    BestCheckpointCallback,
    ExhaustiveValLauncher,
    SigtermHandler,
    TrainMetricsCSV,
    VENACheckpointCallback,
)
from vena.model.fm.maisi.config import TrunkConfig
from vena.model.fm.metrics import RegionSpec

from .runner import generate_run_id, write_provenance

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic schema (training_routine.md §2.3 + ``regions:`` block)
# =============================================================================


class _RunCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str = "s1"
    resume_from: str | None = None
    seed: int = 1337
    device: str = "cuda"
    precision: str = "bf16-mixed"
    full_determinism: bool = False


class _DataCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latents_h5: Path
    fold: int = 0
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    max_train_subjects: int | None = None
    max_val_subjects: int | None = None


class _ControlNetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conditioning_inputs: list[str]
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    perturb_keys: list[str] = Field(default_factory=lambda: ["wt"])


class _TrunkCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: Path
    arch_json: Path | None = None
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    class_token: int = 9
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)


class _ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trunk: _TrunkCfg
    controlnet: _ControlNetCfg
    vae_checkpoint: Path | None = None  # only needed when val image metrics are on


class _RFlowCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_train_timesteps: int = 1000
    use_discrete_timesteps: bool = True
    sample_method: str = "uniform"


class _OptimCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    scheduler: str = "polynomial"


class _EMACfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decay: float = 0.9999
    update_after_step: int = 0
    update_every: int = 1
    inv_gamma: float = 10.0
    power: float = 1.0
    min_value: float = 0.0


class _TrainingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_steps: int = 50_000
    batch_size: int = 4
    grad_accum: int = 1
    checkpoint_every_epochs: int = 5
    log_train_every_steps: int = 100
    best_metric_name: str = "mse_latent"
    best_metric_region: str = "bg"
    best_metric_nfe: int = 5
    gradient_clip_val: float = 1.0


class _ValidationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every_epochs: int = 1
    per_epoch_nfe: int = 5
    full_sweep_every_epochs: int = 5
    sweep_nfes: list[int] = Field(default_factory=lambda: [1, 2, 5, 10, 50])
    qualitative_every_epochs: int = 10
    image_metrics: bool = True  # master switch for image-space PSNR/SSIM
    # Image-space metrics are expensive (one VAE decode/patient) and only
    # meaningful at the canonical per_epoch_nfe, so they run on a slow cadence
    # rather than every epoch. 0 disables them entirely.
    image_metrics_every_epochs: int = 20


class _ExhaustiveValCfg(BaseModel):
    """Asynchronous image-space validation offloaded to a second GPU.

    On a slow cadence the trainer snapshots the EMA weights and launches a
    standalone subprocess (``routines.fm.exhaustive_val``) on ``device`` while
    training continues uninterrupted on the primary GPU. The subprocess samples
    each validation patient at every ``nfe_levels`` entry, decodes to image
    space, compares against the real T1c (percentile-normalised exactly as the
    encoder's input), and writes metrics/timing/figures + ``latent_preds.h5``.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    every_epochs: int = 20
    n_patients: int = 20
    nfe_levels: list[int] = Field(default_factory=lambda: [1, 10, 50])
    image_h5: Path | None = None
    device: str = "cuda:1"
    # Python used to launch the subprocess; defaults to the running interpreter.
    python_executable: str | None = None


class _OutputCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiments_root: Path
    retention_n_checkpoints: int = 3
    tensorboard: bool = False
    wandb: bool = False


class FMTrainRoutineConfig(BaseModel):
    """Pydantic root config for ``vena-fm-train`` v2.

    Loaded via :meth:`from_yaml`; the original YAML is round-tripped into
    ``experiments/{run_id}/config.original.yaml``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: _RunCfg
    data: _DataCfg
    model: _ModelCfg
    loss: dict[str, Any] = Field(default_factory=dict)
    rflow: _RFlowCfg = Field(default_factory=_RFlowCfg)
    optim: _OptimCfg = Field(default_factory=_OptimCfg)
    ema: _EMACfg = Field(default_factory=_EMACfg)
    training: _TrainingCfg = Field(default_factory=_TrainingCfg)
    validation: _ValidationCfg = Field(default_factory=_ValidationCfg)
    exhaustive_val: _ExhaustiveValCfg = Field(default_factory=_ExhaustiveValCfg)
    output: _OutputCfg
    # Region specs are no longer consumed in-process (validation is offloaded),
    # but the field is kept (optional) for backward compatibility with configs
    # that still declare it.
    regions: dict[str, RegionSpec] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> FMTrainRoutineConfig:
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# =============================================================================
# Engine
# =============================================================================


class FMTrainRoutineEngine:
    """End-to-end training engine following training_routine.md."""

    def __init__(self, cfg: FMTrainRoutineConfig, config_yaml_path: Path | None = None) -> None:
        self.cfg = cfg
        self.config_yaml_path = config_yaml_path

    def _make_run_dir(self) -> tuple[str, Path]:
        run_id = generate_run_id(self.cfg.run.stage)
        run_dir = Path(self.cfg.output.experiments_root) / run_id
        # ``qualitative`` and ``performance`` are no longer produced in-process —
        # their content (qualitative figures + latent preds, per-NFE timing) now
        # lives under ``exhaustive_val/epoch_NNN/`` (created by the launcher).
        for sub in ("checkpoints", "logs", "metrics"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def _attach_file_log(self, run_dir: Path) -> logging.Handler:
        """Tee log records to ``logs/train.log`` so the run is self-contained.

        The CLI configures a console (rich) handler; here we add a plain
        file handler on the root logger so the run directory captures its own
        training log regardless of how stdout is redirected.
        """
        handler = logging.FileHandler(run_dir / "logs" / "train.log", mode="a")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        return handler

    def _write_static_provenance(self, run_dir: Path) -> None:
        merged = self.cfg.model_dump(mode="json")
        (run_dir / "config.yaml").write_text(yaml.safe_dump(merged, sort_keys=False))
        if self.config_yaml_path is not None:
            shutil.copy2(self.config_yaml_path, run_dir / "config.original.yaml")
        else:
            (run_dir / "config.original.yaml").write_text(yaml.safe_dump(merged, sort_keys=False))
        write_provenance(run_dir, repo=Path(__file__).resolve().parents[3])

    def _build_exhaustive_job_base(self, cfg: FMTrainRoutineConfig) -> dict[str, Any]:
        """Static fields for the exhaustive-val job YAML (epoch/snapshot added later).

        All paths are stringified so the launcher can ``yaml.safe_dump`` them.
        """
        ev = cfg.exhaustive_val
        if ev.image_h5 is None:
            raise ValueError("exhaustive_val.enabled is true but exhaustive_val.image_h5 is null")
        if cfg.model.vae_checkpoint is None:
            raise ValueError("exhaustive_val.enabled is true but model.vae_checkpoint is null")
        return {
            "stage": cfg.run.stage,
            "seed": cfg.run.seed,
            "trunk": {
                "checkpoint": str(cfg.model.trunk.checkpoint),
                "arch_json": str(cfg.model.trunk.arch_json) if cfg.model.trunk.arch_json else None,
                "arch_overrides": dict(cfg.model.trunk.arch_overrides),
                "class_token": cfg.model.trunk.class_token,
                "spacing_mm": list(cfg.model.trunk.spacing_mm),
            },
            "controlnet": {
                "conditioning_inputs": list(cfg.model.controlnet.conditioning_inputs),
                "arch_overrides": dict(cfg.model.controlnet.arch_overrides),
            },
            "vae_checkpoint": str(cfg.model.vae_checkpoint),
            "rflow": cfg.rflow.model_dump(),
            "ema": cfg.ema.model_dump(),
            "latents_h5": str(cfg.data.latents_h5),
            "image_h5": str(ev.image_h5),
            "fold": cfg.data.fold,
            "nfe_levels": list(ev.nfe_levels),
            "n_patients": ev.n_patients,
        }

    def _resolve_resume_ckpt(self) -> str | None:
        rf = self.cfg.run.resume_from
        if not rf:
            return None
        if rf == "latest":
            root = Path(self.cfg.output.experiments_root)
            latest_dir = max(root.glob("*/"), default=None, key=lambda p: p.stat().st_mtime)
            if latest_dir is None:
                logger.warning("resume_from=latest but no run dir found under %s", root)
                return None
            cand = list((latest_dir / "checkpoints").glob("last.ckpt"))
            if cand:
                logger.info("Resuming from %s", cand[0])
                return str(cand[0])
            cand = sorted((latest_dir / "checkpoints").glob("ema_epoch_*.ckpt"))
            if cand:
                logger.info("Resuming from %s", cand[-1])
                return str(cand[-1])
        elif rf == "best":
            root = Path(self.cfg.output.experiments_root)
            latest_dir = max(root.glob("*/"), default=None, key=lambda p: p.stat().st_mtime)
            if latest_dir is not None:
                cand = list((latest_dir / "checkpoints").glob("ema_best.ckpt"))
                if cand:
                    return str(cand[0])
        else:
            p = Path(rf)
            if p.is_file():
                return str(p)
        logger.warning("resume_from=%r could not be resolved; starting fresh.", rf)
        return None

    def run(self) -> Path:
        cfg = self.cfg
        pl.seed_everything(cfg.run.seed, workers=True)

        run_id, run_dir = self._make_run_dir()
        self._attach_file_log(run_dir)
        self._write_static_provenance(run_dir)
        logger.info("FM-train run_id=%s dir=%s", run_id, run_dir)

        # Decision JSON for downstream consumers.
        (run_dir / "decision.json").write_text(
            json.dumps(
                {
                    "schema_version": "0.2.0",
                    "produced_at": now_iso_utc(),
                    "producer": "routines.fm.train:0.2.0",
                    "run_id": run_id,
                    "stage": cfg.run.stage,
                },
                indent=2,
            )
        )

        # Data. In-process validation is offloaded to the async second-GPU job
        # (see ExhaustiveValLauncher), so the training process runs *only*
        # training on the primary GPU; ``region_resolver``/``vae_decoder`` are
        # not needed here.
        dm = LatentH5DataModule(
            h5_path=cfg.data.latents_h5,
            fold=cfg.data.fold,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            max_train_subjects=cfg.data.max_train_subjects,
            max_val_subjects=cfg.data.max_val_subjects,
            seed=cfg.run.seed,
        )

        # LightningModule (training-only: region_resolver/vae_decoder = None).
        trunk_cfg = TrunkConfig(
            checkpoint=cfg.model.trunk.checkpoint,
            arch_json=cfg.model.trunk.arch_json,
            arch_overrides=cfg.model.trunk.arch_overrides,
            class_token=cfg.model.trunk.class_token,
            spacing_mm=cfg.model.trunk.spacing_mm,
        )
        optim_cfg = {
            "lr": cfg.optim.lr,
            "betas": list(cfg.optim.betas),
            "weight_decay": cfg.optim.weight_decay,
            "warmup_steps": cfg.optim.warmup_steps,
            "scheduler": cfg.optim.scheduler,
            "max_steps": cfg.training.total_steps,
        }
        module = FMLightningModule(
            trunk_config=trunk_cfg,
            conditioning_specs=list(cfg.model.controlnet.conditioning_inputs),
            stage=cfg.run.stage.upper() if cfg.run.stage.startswith("s") else cfg.run.stage,
            loss_cfg=cfg.loss,
            perturb_keys=set(cfg.model.controlnet.perturb_keys),
            controlnet_arch_overrides=cfg.model.controlnet.arch_overrides,
            optim_cfg=optim_cfg,
            rflow_cfg=cfg.rflow.model_dump(),
            ema_cfg=cfg.ema.model_dump(),
            region_resolver=None,
            vae_decoder=None,
        )

        # Checkpoint selection (ema_best) is on the epoch-aggregated training
        # loss, since validation is offloaded and runs asynchronously.
        ckpt_monitor = "train/total_epoch"
        callbacks: list[pl.Callback] = [
            VENACheckpointCallback(
                dirpath=run_dir / "checkpoints",
                retention_n_checkpoints=cfg.output.retention_n_checkpoints,
                every_n_epochs=cfg.training.checkpoint_every_epochs,
                monitor_key=ckpt_monitor,
                best_mode="min",
                save_on_train_epoch_end=True,
            ),
            BestCheckpointCallback(
                dirpath=run_dir / "checkpoints",
                monitor_key=ckpt_monitor,
                best_mode="min",
                save_on_train_epoch_end=True,
            ),
            TrainMetricsCSV(out_dir=run_dir / "metrics"),
            SigtermHandler(ckpt_dir=run_dir / "checkpoints", filename="ema_final.ckpt"),
        ]
        if cfg.exhaustive_val.enabled:
            callbacks.append(
                ExhaustiveValLauncher(
                    run_dir=run_dir,
                    run_id=run_id,
                    job_base=self._build_exhaustive_job_base(cfg),
                    every_epochs=cfg.exhaustive_val.every_epochs,
                    device=cfg.exhaustive_val.device,
                    cwd=Path(__file__).resolve().parents[3],
                    python_executable=cfg.exhaustive_val.python_executable,
                )
            )

        # Trainer. We write our own clean metric CSVs, so Lightning's logger is
        # disabled. Validation is fully offloaded -> ``limit_val_batches=0``.
        trainer = pl.Trainer(
            max_steps=cfg.training.total_steps,
            precision=cfg.run.precision,
            devices=1 if cfg.run.device.startswith("cuda") else "auto",
            accelerator="gpu" if cfg.run.device.startswith("cuda") else "auto",
            log_every_n_steps=cfg.training.log_train_every_steps,
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.grad_accum,
            deterministic=cfg.run.full_determinism,
            default_root_dir=str(run_dir),
            logger=False,
            callbacks=callbacks,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )

        ckpt_path = self._resolve_resume_ckpt()
        trainer.fit(model=module, datamodule=dm, ckpt_path=ckpt_path)

        # Final EMA-state dump on graceful exit.
        final_path = run_dir / "checkpoints" / "ema_final.ckpt"
        trainer.save_checkpoint(str(final_path))
        logger.info("FM-train completed; artifact dir: %s", run_dir)
        return run_dir

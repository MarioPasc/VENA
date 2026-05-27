"""Flow-matching training routine engine.

Thin orchestrator: loads YAML → builds Lightning DataModule + Module →
runs ``Trainer.fit`` → writes a self-describing artifact directory.

Artifact layout::

    <artifact_root>/<UTC-timestamp>/
        config.yaml                 # copy of the resolved input YAML
        config_full.yaml            # OmegaConf dump of the merged config
        git_sha.txt
        decision.json               # machine-readable contract for downstream
        lightning_logs/version_0/
            metrics.csv             # CSVLogger output (loss curves)

The routine never validates metrics — it only logs the training loss. Downstream
evaluation routines (PSNR/SSIM/LPIPS) consume the produced checkpoint.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from vena.data.h5.shared import now_iso_utc, resolve_git_sha
from vena.model.fm.lightning import FMLightningModule, LatentH5DataModule
from vena.model.fm.maisi.config import TrunkConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


class _DataCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latents_h5: Path
    fold: int = 0
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    max_train_subjects: int | None = None


class _TrunkCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: Path
    arch_json: Path | None = None
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    class_token: int = 9
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)


class _ControlNetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conditioning_inputs: list[str]
    arch_overrides: dict[str, Any] = Field(default_factory=dict)
    perturb_keys: list[str] = Field(default_factory=lambda: ["wt"])


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
    warmup_steps: int = 100
    scheduler: str = "polynomial"


class _TrainerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = 50
    precision: str = "bf16-mixed"
    devices: int | list[int] = 1
    accelerator: str = "auto"
    strategy: str = "auto"
    log_every_n_steps: int = 1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    deterministic: bool = False


class FMTrainRoutineConfig(BaseModel):
    """Pydantic root config for ``vena-fm-train``.

    Loaded via :meth:`from_yaml`; the original YAML text is round-tripped into
    the artifact directory.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 42
    site: str = "unknown"
    stage: str = "S1"
    artifact_root: Path
    run_name: str | None = None
    data: _DataCfg
    trunk: _TrunkCfg
    controlnet: _ControlNetCfg
    loss: dict[str, Any] = Field(default_factory=dict)
    rflow: _RFlowCfg = Field(default_factory=_RFlowCfg)
    optim: _OptimCfg = Field(default_factory=_OptimCfg)
    trainer: _TrainerCfg = Field(default_factory=_TrainerCfg)

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
    """End-to-end training routine."""

    def __init__(self, cfg: FMTrainRoutineConfig) -> None:
        self.cfg = cfg

    def _make_artifact_dir(self) -> Path:
        utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        slug = self.cfg.run_name or f"{self.cfg.stage}-fold{self.cfg.data.fold}"
        out = self.cfg.artifact_root / utc / slug
        out.mkdir(parents=True, exist_ok=True)
        latest = self.cfg.artifact_root / "LATEST"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(out)
        except OSError as exc:
            logger.warning("could not refresh LATEST symlink: %s", exc)
        return out

    def _write_provenance(self, out_dir: Path) -> None:
        # Dump the merged/validated config (Pydantic → dict → YAML).
        merged = self.cfg.model_dump(mode="json")
        with (out_dir / "config_full.yaml").open("w") as f:
            yaml.safe_dump(merged, f, sort_keys=False)
        # Git SHA (best-effort).
        sha = resolve_git_sha() or "unknown"
        (out_dir / "git_sha.txt").write_text(sha + "\n")
        # Decision JSON.
        decision = {
            "schema_version": "0.1.0",
            "produced_at": now_iso_utc(),
            "producer": "routines.fm.train:0.1.0",
            "stage": self.cfg.stage,
            "site": self.cfg.site,
            "git_sha": sha,
        }
        (out_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    def run(self) -> Path:
        cfg = self.cfg
        pl.seed_everything(cfg.seed, workers=True)
        out_dir = self._make_artifact_dir()
        self._write_provenance(out_dir)
        logger.info("FM-train artifact dir: %s", out_dir)

        # DataModule.
        dm = LatentH5DataModule(
            h5_path=cfg.data.latents_h5,
            fold=cfg.data.fold,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            max_train_subjects=cfg.data.max_train_subjects,
            seed=cfg.seed,
        )

        # LightningModule.
        trunk_cfg = TrunkConfig(
            checkpoint=cfg.trunk.checkpoint,
            arch_json=cfg.trunk.arch_json,
            arch_overrides=cfg.trunk.arch_overrides,
            class_token=cfg.trunk.class_token,
            spacing_mm=cfg.trunk.spacing_mm,
        )
        optim_cfg = {
            "lr": cfg.optim.lr,
            "betas": list(cfg.optim.betas),
            "weight_decay": cfg.optim.weight_decay,
            "warmup_steps": cfg.optim.warmup_steps,
            "scheduler": cfg.optim.scheduler,
            "max_steps": cfg.trainer.max_steps,
        }
        rflow_cfg = cfg.rflow.model_dump()
        module = FMLightningModule(
            trunk_config=trunk_cfg,
            conditioning_specs=list(cfg.controlnet.conditioning_inputs),
            stage=cfg.stage,
            loss_cfg=cfg.loss,
            perturb_keys=set(cfg.controlnet.perturb_keys),
            controlnet_arch_overrides=cfg.controlnet.arch_overrides,
            optim_cfg=optim_cfg,
            rflow_cfg=rflow_cfg,
        )

        # Logger & callbacks.
        csv_logger = CSVLogger(save_dir=str(out_dir), name="lightning_logs", version=0)
        callbacks = [LearningRateMonitor(logging_interval="step")]

        trainer_kwargs: dict[str, Any] = {
            "max_steps": cfg.trainer.max_steps,
            "precision": cfg.trainer.precision,
            "devices": cfg.trainer.devices,
            "accelerator": cfg.trainer.accelerator,
            "strategy": cfg.trainer.strategy,
            "log_every_n_steps": cfg.trainer.log_every_n_steps,
            "gradient_clip_val": cfg.trainer.gradient_clip_val,
            "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
            "deterministic": cfg.trainer.deterministic,
            "default_root_dir": str(out_dir),
            "logger": csv_logger,
            "callbacks": callbacks,
            "enable_checkpointing": False,  # smoke: no checkpoints written
            "enable_progress_bar": True,
            "enable_model_summary": True,
        }
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model=module, datamodule=dm)

        logger.info("FM-train completed; artifact dir: %s", out_dir)
        return out_dir

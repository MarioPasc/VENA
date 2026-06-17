"""Engine for the 3D-LDDPM competitor benchmark routine.

Pydantic config → run_id → preflight → decision.json (competitor schema 1.0
with a ``competitor.deviations`` extension block) → ``train_lddpm_3d``.

Citation
--------
Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models."
*NeurIPS 2020*. arXiv:2006.11239.

Eidex, Z. *et al.* 2025. "An Efficient 3D Latent Diffusion Model for
T1-contrast Enhanced MRI Generation." arXiv:2509.24194. (§4 baseline
recipe — DDPM scheduler + MSE epsilon loss + MAISI latents.)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


COMPETITOR_NAME = "lddpm_3d"
COMPETITOR_PAPER = "Ho et al. 2020 (DDPM) + Eidex et al. 2025 §4 (3D + MAISI latents recipe)"
COMPETITOR_DOI = "arXiv:2006.11239; arXiv:2509.24194"
COMPETITOR_REPO = (
    "https://github.com/zacheidex/"
    "An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation"
)
PRODUCER_VERSION = "0.1.0"


def _read_upstream_sha() -> str:
    """Return the vendored upstream SHA from ``src/external/lddpm_3d/UPSTREAM_SHA.txt``."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    sha_file = repo_root / "src" / "external" / "lddpm_3d" / "UPSTREAM_SHA.txt"
    if sha_file.is_file():
        return sha_file.read_text().strip()
    return "unknown"


def _short_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class DataCfg(BaseModel):
    """Where to read VENA's latent data from (exactly one of {latent_h5, corpus_registry})."""

    latent_h5: Path | None = None
    corpus_registry: Path | None = None
    fold: int = 0
    input_latents: tuple[str, ...] = ("t1pre", "flair")
    target_latent: str = "t1c"
    max_patients_per_cohort: int | None = None
    cohort_path_overrides: dict[str, Path] = Field(default_factory=dict)


class HyperParamsCfg(BaseModel):
    """3D-LDDPM hyperparameters.

    Defaults mirror upstream ``train_ddpm.py`` (SHA fc8314f6) and the
    paper-faithful U-Net backbone shared with the T1C-RFlow wrapper.
    """

    # Backbone (paper-faithful U-Net — same as T1C-RFlow wrapper).
    latent_channels: int = 4
    cond_latents: int = 2  # T1pre + FLAIR

    # DDPM scheduler kwargs (train_ddpm.py:113-119).
    num_train_timesteps: int = 1000
    beta_start: float = 0.0015
    beta_end: float = 0.0195
    beta_schedule: str = "scaled_linear_beta"
    clip_sample: bool = False

    # Optimisation (Eidex 2025 §4 — same as T1C-RFlow runner).
    lr: float = 1.0e-5
    weight_decay: float = 1.0e-4
    batch_size: int = 4

    # Training schedule + early stop — VENA paired-comparison axes.
    max_epochs: int = 10000
    patience: int = 100
    save_epoch_freq: int = 25
    log_every: int = 200
    num_workers: int = 8
    use_amp: bool = True


class RuntimeCfg(BaseModel):
    """Where to put artifacts and which platform we are on."""

    experiments_root: Path
    platform: Literal["server3", "loginexa", "picasso", "local"]
    tag: str
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    seed: int = 1337


class LDDPM3DCompetitorConfig(BaseModel):
    """Top-level config for the 3D-LDDPM competitor routine."""

    runtime: RuntimeCfg
    data: DataCfg
    hp: HyperParamsCfg

    @model_validator(mode="after")
    def _check_consistency(self) -> LDDPM3DCompetitorConfig:
        if self.hp.cond_latents != len(self.data.input_latents):
            raise ValueError(
                f"hp.cond_latents={self.hp.cond_latents} disagrees with "
                f"len(data.input_latents)={len(self.data.input_latents)}"
            )
        if bool(self.data.latent_h5) == bool(self.data.corpus_registry):
            raise ValueError(
                "DataCfg requires exactly one of {latent_h5, corpus_registry} to be set."
            )
        if self.hp.beta_start <= 0 or self.hp.beta_end <= self.hp.beta_start:
            raise ValueError(
                f"DDPM beta schedule invalid: 0 < beta_start ({self.hp.beta_start}) "
                f"< beta_end ({self.hp.beta_end}) is required."
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> LDDPM3DCompetitorConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LDDPM3DCompetitorEngine:
    """Thin orchestrator — Pydantic cfg → run_dir → train_lddpm_3d → decision.json."""

    def __init__(
        self,
        cfg: LDDPM3DCompetitorConfig,
        config_yaml_path: Path,
    ) -> None:
        self.cfg = cfg
        self.config_yaml_path = Path(config_yaml_path).resolve()
        self.repo_root = Path(__file__).resolve().parents[3]

    def _generate_run_id(self) -> str:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        sha = _short_git_sha(self.repo_root)
        return f"{ts}_competitor_lddpm_3d_{self.cfg.runtime.tag}_{sha}"

    def _build_runner_cfg(self) -> SimpleNamespace:
        d, h = self.cfg.data, self.cfg.hp
        gpu_id = self.cfg.runtime.gpu_ids[0] if self.cfg.runtime.gpu_ids else 0
        return SimpleNamespace(
            corpus_registry=d.corpus_registry,
            latent_h5=d.latent_h5,
            cohort_path_overrides=d.cohort_path_overrides,
            max_patients_per_cohort=d.max_patients_per_cohort,
            fold=d.fold,
            input_latents=d.input_latents,
            target_latent=d.target_latent,
            latent_channels=h.latent_channels,
            cond_latents=h.cond_latents,
            num_train_timesteps=h.num_train_timesteps,
            beta_start=h.beta_start,
            beta_end=h.beta_end,
            beta_schedule=h.beta_schedule,
            clip_sample=h.clip_sample,
            lr=h.lr,
            weight_decay=h.weight_decay,
            batch_size=h.batch_size,
            max_epochs=h.max_epochs,
            patience=h.patience,
            save_epoch_freq=h.save_epoch_freq,
            log_every=h.log_every,
            num_workers=h.num_workers,
            use_amp=h.use_amp,
            gpu_id=gpu_id,
            seed=self.cfg.runtime.seed,
        )

    def _write_decision(self, run_dir: Path, completed: bool) -> None:
        decision = {
            "schema_version": "1.0",
            "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "producer": f"routines.competitors.lddpm_3d:{PRODUCER_VERSION}",
            "completed": completed,
            "competitor": {
                "name": COMPETITOR_NAME,
                "paper": COMPETITOR_PAPER,
                "doi": COMPETITOR_DOI,
                "upstream_repo": COMPETITOR_REPO,
                "upstream_sha": _read_upstream_sha(),
                "deviations": {
                    "backbone": (
                        "MAISI U-Net (paper-faithful) — num_channels="
                        "[128, 128, 256], 3 levels, 2 res-blocks per level, "
                        "no self-attention. Identical to the T1C-RFlow "
                        "wrapper backbone — this isolates scheduler/loss as "
                        "the only competitor-internal delta vs T1C-RFlow."
                    ),
                    "scheduler": (
                        "MONAI DDPMScheduler — num_train_timesteps=1000, "
                        "beta_start=0.0015, beta_end=0.0195, "
                        "schedule='scaled_linear_beta', clip_sample=False. "
                        "Mirrors upstream train_ddpm.py:113-119 exactly. "
                        "Departure from T1C-RFlow's RFlow scheduler is "
                        "intentional — Eidex 2025 §4 reports LDDPM as a "
                        "baseline of the headline RFlow method."
                    ),
                    "loss": (
                        "MSE on noise prediction "
                        "(F.mse_loss(eps_pred, eps)) — standard Ho et al. "
                        "2020 epsilon-prediction loss; mirrors upstream "
                        "train_ddpm.py:170. Not L1 on velocity."
                    ),
                    "conditioning": (
                        "channel-wise concat through MONAI DiffusionInferer "
                        "with mode='concat' — [noisy_z_T1c, z_T1pre, "
                        "z_FLAIR]; in_channels = latent_channels * 3 = 12. "
                        "Generalises upstream train_ddpm.py:103 (single-"
                        "condition × 2) to two conditions (× 3)."
                    ),
                    "augmentation": (
                        "none — dataset reads cohort.latent_h5 only; deterministic by contract."
                    ),
                    "vae_handling": (
                        "vena_maisi_v2_symmetric_baseline — same MAISI-V2 "
                        "latents both for VENA and this competitor; the VAE "
                        "is symmetric across the comparison so the VAE-choice "
                        "confound does not contaminate VENA-vs-3D-LDDPM "
                        "deltas. Upstream's autoencoder_epoch273.pt is NOT "
                        "loaded."
                    ),
                    "intensity_norm_for_metrics": (
                        "VENA percentile_normalise(99.5, foreground_only) "
                        "for the real-T1c reference at inference time. "
                        "Decoded predictions live in the MAISI [0, 1] space "
                        "natively."
                    ),
                    "ema": "none (paper and wrapper)",
                    "amp": (
                        "enabled by default — upstream train_ddpm.py uses "
                        "GradScaler + autocast(fp16)."
                    ),
                    "t2_flair_substitute": (
                        "FLAIR only — UCSF-PDGM has T2 and FLAIR separately; "
                        "Eidex 2025's 'T2-FLAIR' is BraTS-combined. We use "
                        "FLAIR as the closest 1:1 substitute, same as "
                        "T1C-RFlow."
                    ),
                    "beta_start_inference": (
                        "uses training-time beta_start=0.0015 (not the "
                        "upstream test_ddpm_t1_flair_final.py:125 value of "
                        "0.0005, which looks like a typo). A sampler that "
                        "does not match the training schedule produces a "
                        "different noise profile than the model was trained "
                        "under."
                    ),
                },
            },
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "tag": self.cfg.runtime.tag,
            "platform": self.cfg.runtime.platform,
            "seed": self.cfg.runtime.seed,
            "git_sha": _short_git_sha(self.repo_root),
            "data": {
                "latent_h5": str(self.cfg.data.latent_h5) if self.cfg.data.latent_h5 else None,
                "corpus_registry": str(self.cfg.data.corpus_registry)
                if self.cfg.data.corpus_registry
                else None,
                "cohort_path_overrides": {
                    k: str(v) for k, v in self.cfg.data.cohort_path_overrides.items()
                },
                "fold": self.cfg.data.fold,
                "input_latents": list(self.cfg.data.input_latents),
                "target_latent": self.cfg.data.target_latent,
                "max_patients_per_cohort": self.cfg.data.max_patients_per_cohort,
            },
            "hyperparams": json.loads(self.cfg.hp.model_dump_json()),
            "runtime": {
                "gpu_ids": self.cfg.runtime.gpu_ids,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    def _preflight(self) -> None:
        if self.cfg.data.latent_h5 is not None and not self.cfg.data.latent_h5.is_file():
            raise FileNotFoundError(f"latent_h5 missing: {self.cfg.data.latent_h5}")
        if (
            self.cfg.data.corpus_registry is not None
            and not self.cfg.data.corpus_registry.is_file()
        ):
            raise FileNotFoundError(f"corpus_registry missing: {self.cfg.data.corpus_registry}")
        # Vendored upstream must be present (reference files; the wrapper does
        # not import from them but we keep the structural sanity check).
        upstream_dir = self.repo_root / "src" / "external" / "lddpm_3d" / "upstream"
        if not (upstream_dir / "train_ddpm.py").is_file():
            raise FileNotFoundError(
                f"vendored LDDPM upstream missing at {upstream_dir}; "
                "re-vendor per src/external/lddpm_3d/UPSTREAM.md"
            )
        anchor = self.cfg.runtime.experiments_root.parent.parent
        if not anchor.exists():
            raise FileNotFoundError(f"experiments_root grandparent does not exist: {anchor}")
        self.cfg.runtime.experiments_root.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        self._preflight()

        import random as _random

        import numpy as _np
        import torch as _torch

        seed = self.cfg.runtime.seed
        _random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)

        run_id = self._generate_run_id()
        run_dir = self.cfg.runtime.experiments_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self.config_yaml_path, run_dir / "config.original.yaml")
        (run_dir / "config.resolved.json").write_text(
            json.dumps(
                json.loads(self.cfg.model_dump_json()),
                indent=2,
                default=str,
            )
        )
        self._write_decision(run_dir, completed=False)
        logger.info("run_dir = %s", run_dir)

        from vena.competitors.lddpm_3d import train_lddpm_3d

        train_lddpm_3d(self._build_runner_cfg(), run_dir)

        self._write_decision(run_dir, completed=True)
        return run_dir

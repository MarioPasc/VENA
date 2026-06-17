"""Engine for the 3D-Latent-Pix2Pix competitor benchmark routine.

Pydantic config → run_id → preflight → decision.json (competitor schema 1.0
with a ``competitor.deviations`` extension block) → ``train_lpix2pix_3d``.

Citation
--------
Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. "Image-to-Image Translation
with Conditional Adversarial Networks." *CVPR 2017*. arXiv:1611.07004.

Eidex, Z. *et al.* 2025. "An Efficient 3D Latent Diffusion Model for
T1-contrast Enhanced MRI Generation." arXiv:2509.24194. (§4 "Pix2pix"
baseline — same MAISI-V2 latents + channel-concat conditioning.)
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


COMPETITOR_NAME = "lpix2pix_3d"
COMPETITOR_PAPER = (
    "Isola et al. 2017 (Pix2Pix conditional GAN) + Eidex et al. 2025 §4 "
    "(3D-latent baseline over MAISI-V2)"
)
COMPETITOR_DOI = "arXiv:1611.07004; arXiv:2509.24194"
COMPETITOR_REPO = (
    "https://github.com/zacheidex/"
    "An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation"
)
PRODUCER_VERSION = "0.1.0"


def _read_upstream_sha() -> str:
    """Return the vendored upstream SHA from ``src/external/lpix2pix_3d/UPSTREAM_SHA.txt``."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    sha_file = repo_root / "src" / "external" / "lpix2pix_3d" / "UPSTREAM_SHA.txt"
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
    """3D-Latent-Pix2Pix hyperparameters.

    Defaults are the Isola *et al.* 2017 Pix2Pix recipe (BCE + λ·L1 with
    λ=100, AdamW β1=0.5) instantiated on VENA's MAISI-V2 latent grid
    ``(4, 48, 56, 48)`` with two conditioning modalities (T1pre + FLAIR).
    Optimiser kwargs follow the vendored upstream
    ``train_pix2pix_t1n_t2f.py`` (lr=1e-4, β2=0.999, wd=1e-4), which
    matches modern best practice and was the configuration Eidex *et al.*
    2025 evaluated.
    """

    # Generator architecture — paper-faithful MAISI 3-level U-Net (same as
    # T1C-RFlow). No knobs exposed; the architecture is locked.
    latent_channels: int = 4
    cond_latents: int = 2  # T1pre + FLAIR

    # PatchGAN discriminator (Isola §6.1).
    disc_ndf: int = 64
    disc_num_layers: int = 4

    # Pix2Pix loss + optimisation (Isola §3.2 + vendored train script).
    lambda_l1: float = 100.0
    lr_g: float = 1.0e-4
    lr_d: float = 1.0e-4
    beta1: float = 0.5
    beta2: float = 0.999
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


class LPix2Pix3DCompetitorConfig(BaseModel):
    """Top-level config for the 3D-Latent-Pix2Pix competitor routine."""

    runtime: RuntimeCfg
    data: DataCfg
    hp: HyperParamsCfg

    @model_validator(mode="after")
    def _check_consistency(self) -> LPix2Pix3DCompetitorConfig:
        if self.hp.cond_latents != len(self.data.input_latents):
            raise ValueError(
                f"hp.cond_latents={self.hp.cond_latents} disagrees with "
                f"len(data.input_latents)={len(self.data.input_latents)}"
            )
        if bool(self.data.latent_h5) == bool(self.data.corpus_registry):
            raise ValueError(
                "DataCfg requires exactly one of {latent_h5, corpus_registry} to be set."
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> LPix2Pix3DCompetitorConfig:
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LPix2Pix3DCompetitorEngine:
    """Thin orchestrator — Pydantic cfg → run_dir → train_lpix2pix_3d → decision.json."""

    def __init__(
        self,
        cfg: LPix2Pix3DCompetitorConfig,
        config_yaml_path: Path,
    ) -> None:
        self.cfg = cfg
        self.config_yaml_path = Path(config_yaml_path).resolve()
        self.repo_root = Path(__file__).resolve().parents[3]

    def _generate_run_id(self) -> str:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        sha = _short_git_sha(self.repo_root)
        return f"{ts}_competitor_lpix2pix_3d_{self.cfg.runtime.tag}_{sha}"

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
            disc_ndf=h.disc_ndf,
            disc_num_layers=h.disc_num_layers,
            lambda_l1=h.lambda_l1,
            lr_g=h.lr_g,
            lr_d=h.lr_d,
            beta1=h.beta1,
            beta2=h.beta2,
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
            "producer": f"routines.competitors.lpix2pix_3d:{PRODUCER_VERSION}",
            "completed": completed,
            "competitor": {
                "name": COMPETITOR_NAME,
                "paper": COMPETITOR_PAPER,
                "doi": COMPETITOR_DOI,
                "upstream_repo": COMPETITOR_REPO,
                "upstream_sha": _read_upstream_sha(),
                "deviations": {
                    "generator_backbone": (
                        "DiffusionModelUNetMaisi at the paper-faithful "
                        "3-level config (num_channels=[128, 128, 256], "
                        "num_res_blocks=2, no self-attention) — same as the "
                        "T1C-RFlow wrapper. Vendored code uses the 4-level "
                        "+ attention config_maisi3d-rflow.json (178 M params); "
                        "we adopt the 3-level paper-faithful variant so the "
                        "only axis isolated against T1C-RFlow is the training "
                        "paradigm (GAN vs flow-matching)."
                    ),
                    "generator_wrapper": (
                        "GeneratorUNetWrapper(unet) — feeds zero timesteps "
                        "(t = zeros((B,), dtype=long)) so the diffusion U-Net "
                        "runs as a deterministic conditional generator. One "
                        "forward pass per prediction; no scheduler, no noise."
                    ),
                    "discriminator": (
                        "PatchDiscriminator3D — 4 strided Conv3d + "
                        "InstanceNorm + LeakyReLU(0.2) layers, ndf=64, "
                        "terminal 1-channel patch-logits head. Receptive field "
                        "matches Isola §6.1 70x70 PatchGAN in 3D adaptation."
                    ),
                    "loss": (
                        "BCEWithLogitsLoss adversarial + lambda_l1 * "
                        "F.l1_loss(fake, real) with lambda_l1=100 — Isola "
                        "2017 §3.2 verbatim; vendored code matches."
                    ),
                    "conditioning": (
                        "channel-wise concat — Generator sees "
                        "[z_T1pre, z_FLAIR] (8 channels); Discriminator sees "
                        "[cond, target_or_fake] (12 channels)."
                    ),
                    "augmentation": (
                        "none — dataset reads cohort.latent_h5 only; deterministic by contract."
                    ),
                    "vae_handling": (
                        "vena_maisi_v2_symmetric_baseline — same MAISI-V2 "
                        "latents both for VENA and this competitor; the VAE "
                        "is symmetric across the comparison so the VAE-choice "
                        "confound does not contaminate VENA-vs-Pix2Pix deltas."
                    ),
                    "intensity_norm_for_metrics": (
                        "VENA percentile_normalise(99.5, foreground_only) "
                        "for the real-T1c reference at inference time. "
                        "Decoded predictions live in the MAISI [0, 1] space "
                        "natively."
                    ),
                    "ema": "none (paper and wrapper)",
                    "amp": (
                        "enabled by default — vendored code uses GradScaler + "
                        "autocast(fp16) at every G/D step; same posture as "
                        "T1C-RFlow / DiT-3D wrappers."
                    ),
                    "optimiser": (
                        "AdamW(lr_G=lr_D=1e-4, betas=(0.5, 0.999), wd=1e-4) "
                        "for both networks. Departure from Isola 2017's "
                        "Adam(lr=2e-4) is in the vendored code and matches "
                        "Eidex 2025's baseline configuration."
                    ),
                    "model_selection_metric": (
                        "epoch-mean loss_g_l1 — the L1 reconstruction "
                        "component is the stable signal; total G loss is "
                        "dominated by the BCE term which oscillates against "
                        "D. Same metric Isola §6 reports."
                    ),
                    "t2_flair_substitute": (
                        "FLAIR only — UCSF-PDGM has T2 and FLAIR separately; "
                        "Eidex 2025's 'T2-FLAIR' is BraTS-combined. We use "
                        "FLAIR as the closest 1:1 substitute, same as "
                        "T1C-RFlow and DiT-3D wrappers."
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
        # Vendored upstream metadata must be present (decision.json reads the SHA).
        upstream_dir = self.repo_root / "src" / "external" / "lpix2pix_3d"
        if not (upstream_dir / "UPSTREAM_SHA.txt").is_file():
            raise FileNotFoundError(
                f"vendored 3D-Latent-Pix2Pix upstream metadata missing at "
                f"{upstream_dir}; re-vendor per "
                "src/external/lpix2pix_3d/UPSTREAM.md"
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

        from vena.competitors.lpix2pix_3d import train_lpix2pix_3d

        train_lpix2pix_3d(self._build_runner_cfg(), run_dir)

        self._write_decision(run_dir, completed=True)
        return run_dir

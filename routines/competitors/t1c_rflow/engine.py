"""Engine for the T1C-RFlow competitor benchmark routine.

Wires a Pydantic config to :func:`vena.competitors.t1c_rflow.train_t1c_rflow`,
generates a deterministic run id, writes ``decision.json`` (competitor schema
1.0 with a ``competitor.deviations`` extension block), and returns the run
directory path.

Citation
--------
Eidex *et al.* 2025, "An Efficient 3D Latent Diffusion Model for T1-contrast
Enhanced MRI Generation," arXiv:2509.24194.
"""

from __future__ import annotations

import hashlib
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


COMPETITOR_NAME = "t1c_rflow"
COMPETITOR_PAPER = "Eidex et al. 2025, arXiv:2509.24194"
COMPETITOR_DOI = "arXiv:2509.24194"
COMPETITOR_REPO = (
    "https://github.com/zacheidex/"
    "An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation"
)
PRODUCER_VERSION = "0.1.0"


def _read_upstream_sha() -> str:
    """Return the vendored upstream SHA from ``src/external/t1c_rflow/UPSTREAM_SHA.txt``."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    sha_file = repo_root / "src" / "external" / "t1c_rflow" / "UPSTREAM_SHA.txt"
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


def _vendored_arch_config_default() -> Path:
    """Return the vendored ``maisi/configs/config_maisi3d-rflow.json`` path."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return (
        repo_root
        / "src" / "external" / "t1c_rflow" / "upstream"
        / "maisi" / "configs" / "config_maisi3d-rflow.json"
    )


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class DataCfg(BaseModel):
    """Where to read VENA's latent data from.

    Two modes (exactly one must be set):

    1. **Single-cohort** (smoke / sanity): ``latent_h5`` points at one
       VENA-produced latent H5.
    2. **Multi-cohort** (production / fair comparison vs VENA):
       ``corpus_registry`` points at a VENA corpus-registry JSON; the runner
       concatenates every cohort with ``role="cv"``.
    """

    latent_h5: Path | None = None
    corpus_registry: Path | None = None
    fold: int = 0
    input_latents: tuple[str, ...] = ("t1pre", "flair")
    target_latent: str = "t1c"
    max_patients_per_cohort: int | None = None
    # Per-platform overrides keyed by cohort name (e.g. "UCSF-PDGM").
    cohort_path_overrides: dict[str, Path] = Field(default_factory=dict)


class HyperParamsCfg(BaseModel):
    """T1C-RFlow hyperparameters. Defaults match Eidex et al. 2025 §4."""

    # Architecture (from vendored maisi/configs/config_maisi3d-rflow.json).
    unet_arch_config: Path = Field(default_factory=_vendored_arch_config_default)
    latent_channels: int = 4
    cond_latents: int = 2  # T1pre + FLAIR

    # RFlow scheduler.
    nfe_train_timesteps: int = 1000

    # Optimisation (paper §4).
    lr: float = 1.0e-5
    weight_decay: float = 1.0e-4
    batch_size: int = 4

    # Training schedule + early stop. ``max_epochs`` and ``patience`` are the
    # VENA paired-comparison axes (mirror ``picasso_s1_1000ep_fft.yaml``).
    max_epochs: int = 10000
    patience: int = 100
    save_epoch_freq: int = 25
    log_every: int = 200
    num_workers: int = 8

    # AMP enabled by default (upstream ``train_rflow.py`` uses it).
    use_amp: bool = True


class RuntimeCfg(BaseModel):
    """Where to put artifacts and which platform we are on."""

    experiments_root: Path
    platform: Literal["server3", "loginexa", "picasso", "local"]
    tag: str
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    seed: int = 1337


class T1CRFlowCompetitorConfig(BaseModel):
    """Top-level config for the T1C-RFlow competitor routine."""

    runtime: RuntimeCfg
    data: DataCfg
    hp: HyperParamsCfg

    @model_validator(mode="after")
    def _check_consistency(self) -> "T1CRFlowCompetitorConfig":
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
    def from_yaml(cls, path: Path) -> "T1CRFlowCompetitorConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class T1CRFlowCompetitorEngine:
    """Thin orchestrator — Pydantic cfg → run_dir → train_t1c_rflow → decision.json."""

    def __init__(
        self,
        cfg: T1CRFlowCompetitorConfig,
        config_yaml_path: Path,
    ) -> None:
        self.cfg = cfg
        self.config_yaml_path = Path(config_yaml_path).resolve()
        self.repo_root = Path(__file__).resolve().parents[3]

    def _generate_run_id(self) -> str:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        sha = _short_git_sha(self.repo_root)
        return f"{ts}_competitor_t1c_rflow_{self.cfg.runtime.tag}_{sha}"

    def _build_runner_cfg(self) -> SimpleNamespace:
        """Translate Pydantic config → plain SimpleNamespace for ``train_t1c_rflow``."""
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
            unet_arch_config=h.unet_arch_config,
            latent_channels=h.latent_channels,
            cond_latents=h.cond_latents,
            nfe_train_timesteps=h.nfe_train_timesteps,
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

    def _file_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_decision(self, run_dir: Path, completed: bool) -> None:
        decision = {
            "schema_version": "1.0",
            "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "producer": f"routines.competitors.t1c_rflow:{PRODUCER_VERSION}",
            "completed": completed,
            "competitor": {
                "name": COMPETITOR_NAME,
                "paper": COMPETITOR_PAPER,
                "doi": COMPETITOR_DOI,
                "upstream_repo": COMPETITOR_REPO,
                "upstream_sha": _read_upstream_sha(),
                # Explicit list of documented deviations from the paper.
                # Reviewers should read this block first.
                "deviations": {
                    "unet_architecture": (
                        "paper_faithful_3level — wrapper builds [128, 128, 256] "
                        "3-level conv-only U-Net (49.6M params) per Eidex 2025 "
                        "§3 text. Released code uses [64, 128, 256, 512] + "
                        "self-attention at levels 3-4 (178.6M params). VENA "
                        "policy 2026-06-15: follow peer-reviewed text over "
                        "unreviewed code."
                    ),
                    "latent_sampling": (
                        "static_z — VENA's stored z is read directly; paper "
                        "resamples z = mu + sigma * eps per training step. "
                        "Expected effect ≤0.2 dB PSNR (see UPSTREAM.md)."
                    ),
                    "vae_handling": (
                        "vena_maisi_v2_symmetric_baseline — paper cites "
                        "Guo et al. 2024 MAISI; release ships their own "
                        "epoch-273 retraining. We use VENA's MAISI-V2 "
                        "latents for both our model and this competitor, so "
                        "the VAE is symmetric across the comparison."
                    ),
                    "intensity_norm_for_metrics": (
                        "VENA percentile_normalise(99.5, foreground_only) "
                        "vs upstream minmax01. Keeps metrics comparable to "
                        "VENA's own benchmark numbers."
                    ),
                    "augmentation": (
                        "none — wrapper reads only cohort.latent_h5, never "
                        "cohort.latent_aug_h5; dataset is deterministic by "
                        "contract (pinned by test_dataset_is_deterministic)."
                    ),
                    "ema": "none (paper and wrapper)",
                    "amp": (
                        "enabled by default — upstream train_rflow.py uses "
                        "torch.amp.GradScaler + autocast(fp16); paper §4 is "
                        "silent on AMP. Kept because the paper's stated 100-"
                        "epoch / A6000-ADA budget is only feasible with it. "
                        "Toggle via hp.use_amp."
                    ),
                    "t2_flair_substitute": (
                        "FLAIR only (UCSF-PDGM has T2 and FLAIR separately; "
                        "the paper's 'T2-FLAIR' is BraTS combined). "
                        "Substitute = FLAIR — closest 1:1."
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
                "latent_h5": str(self.cfg.data.latent_h5)
                if self.cfg.data.latent_h5 else None,
                "corpus_registry": str(self.cfg.data.corpus_registry)
                if self.cfg.data.corpus_registry else None,
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
            raise FileNotFoundError(
                f"latent_h5 missing: {self.cfg.data.latent_h5}"
            )
        if self.cfg.data.corpus_registry is not None and not self.cfg.data.corpus_registry.is_file():
            raise FileNotFoundError(
                f"corpus_registry missing: {self.cfg.data.corpus_registry}"
            )
        if not self.cfg.hp.unet_arch_config.is_file():
            raise FileNotFoundError(
                f"unet_arch_config missing: {self.cfg.hp.unet_arch_config}"
            )
        anchor = self.cfg.runtime.experiments_root.parent.parent
        if not anchor.exists():
            raise FileNotFoundError(
                f"experiments_root grandparent does not exist: {anchor}"
            )
        self.cfg.runtime.experiments_root.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        self._preflight()

        # Seed reproducibility (the dataset is deterministic by construction;
        # this affects only the UNet init RNG and noise sampling).
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

        # Persist the resolved config.
        shutil.copy2(self.config_yaml_path, run_dir / "config.original.yaml")
        (run_dir / "config.resolved.json").write_text(
            json.dumps(
                json.loads(self.cfg.model_dump_json()),
                indent=2,
                default=str,
            )
        )
        # Preliminary decision.json — partial / crashed runs are still tracked.
        self._write_decision(run_dir, completed=False)
        logger.info("run_dir = %s", run_dir)

        from vena.competitors.t1c_rflow import train_t1c_rflow

        train_t1c_rflow(self._build_runner_cfg(), run_dir)

        # Final decision.json now that training reached the end.
        self._write_decision(run_dir, completed=True)
        return run_dir

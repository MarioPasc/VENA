"""Engine for the 3D-DiT competitor benchmark routine.

Pydantic config → run_id → preflight → decision.json (competitor schema 1.0
with a ``competitor.deviations`` extension block) → ``train_dit_3d``.

Citation
--------
Peebles, W., & Xie, S. "Scalable Diffusion Models with Transformers."
*ICCV 2023*. arXiv:2212.09748.

Eidex, Z. *et al.* 2025. "An Efficient 3D Latent Diffusion Model for
T1-contrast Enhanced MRI Generation." arXiv:2509.24194. (§4 baseline
recipe — RFlow scheduler + L1 velocity loss + MAISI latents.)
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


COMPETITOR_NAME = "dit_3d"
COMPETITOR_PAPER = (
    "Peebles & Xie 2023 (DiT backbone) + Eidex et al. 2025 §4 (3D + MAISI + RFlow recipe)"
)
COMPETITOR_DOI = "arXiv:2212.09748; arXiv:2509.24194"
COMPETITOR_REPO = (
    "https://github.com/zacheidex/"
    "An-Efficient-3D-Latent-Diffusion-Model-for-T1-contrast-Enhanced-MRI-Generation"
)
PRODUCER_VERSION = "0.1.0"


def _read_upstream_sha() -> str:
    """Return the vendored upstream SHA from ``src/external/dit_3d/UPSTREAM_SHA.txt``."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    sha_file = repo_root / "src" / "external" / "dit_3d" / "UPSTREAM_SHA.txt"
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
    """3D-DiT hyperparameters.

    Defaults are the Peebles & Xie 2023 DiT-B configuration with patch_size=4
    (chosen so the multi-cohort latent grid ``(48, 56, 48)`` is cleanly
    divisible), and the Eidex 2025 §4 training recipe (RFlow + L1 velocity).
    """

    # DiT-B/4 in 3D — Peebles & Xie 2023 standard "base" configuration.
    latent_channels: int = 4
    cond_latents: int = 2  # T1pre + FLAIR
    dit_hidden_size: int = 768
    dit_depth: int = 12
    dit_num_heads: int = 12
    dit_patch_size: int = 4
    dit_mlp_ratio: float = 4.0

    # RFlow scheduler.
    nfe_train_timesteps: int = 1000

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


class DiT3DCompetitorConfig(BaseModel):
    """Top-level config for the 3D-DiT competitor routine."""

    runtime: RuntimeCfg
    data: DataCfg
    hp: HyperParamsCfg

    @model_validator(mode="after")
    def _check_consistency(self) -> "DiT3DCompetitorConfig":
        if self.hp.cond_latents != len(self.data.input_latents):
            raise ValueError(
                f"hp.cond_latents={self.hp.cond_latents} disagrees with "
                f"len(data.input_latents)={len(self.data.input_latents)}"
            )
        if bool(self.data.latent_h5) == bool(self.data.corpus_registry):
            raise ValueError(
                "DataCfg requires exactly one of {latent_h5, corpus_registry} to be set."
            )
        if self.hp.dit_hidden_size % 3 != 0:
            raise ValueError(
                f"dit_hidden_size={self.hp.dit_hidden_size} must be divisible "
                "by 3 (the 3D sin-cos positional embedding splits the dim "
                "into thirds — see src/external/dit_3d/upstream/dit3d.py:332)."
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "DiT3DCompetitorConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DiT3DCompetitorEngine:
    """Thin orchestrator — Pydantic cfg → run_dir → train_dit_3d → decision.json."""

    def __init__(
        self,
        cfg: DiT3DCompetitorConfig,
        config_yaml_path: Path,
    ) -> None:
        self.cfg = cfg
        self.config_yaml_path = Path(config_yaml_path).resolve()
        self.repo_root = Path(__file__).resolve().parents[3]

    def _generate_run_id(self) -> str:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        sha = _short_git_sha(self.repo_root)
        return f"{ts}_competitor_dit_3d_{self.cfg.runtime.tag}_{sha}"

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
            dit_hidden_size=h.dit_hidden_size,
            dit_depth=h.dit_depth,
            dit_num_heads=h.dit_num_heads,
            dit_patch_size=h.dit_patch_size,
            dit_mlp_ratio=h.dit_mlp_ratio,
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

    def _write_decision(self, run_dir: Path, completed: bool) -> None:
        decision = {
            "schema_version": "1.0",
            "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "producer": f"routines.competitors.dit_3d:{PRODUCER_VERSION}",
            "completed": completed,
            "competitor": {
                "name": COMPETITOR_NAME,
                "paper": COMPETITOR_PAPER,
                "doi": COMPETITOR_DOI,
                "upstream_repo": COMPETITOR_REPO,
                "upstream_sha": _read_upstream_sha(),
                "deviations": {
                    "backbone": (
                        "DiT-B/4 in 3D (paper-faithful) — depth=12, "
                        "hidden=768, num_heads=12, patch_size=4, mlp_ratio=4.0. "
                        "Built from the vendored DiT3DWrapper at "
                        "src/external/dit_3d/upstream/. Eidex 2025 §4 does "
                        "not pin the DiT-3D size; we match Peebles & Xie "
                        "2023's standard 'base' configuration."
                    ),
                    "scheduler": (
                        "RFlow (rectified flow) — same kwargs as VENA / "
                        "T1C-RFlow: num_train_timesteps=1000, "
                        "use_discrete_timesteps=True, "
                        "sample_method='logit-normal', "
                        "use_timestep_transform=True, "
                        "base_img_size_numel=64*64*48, spatial_dim=3. "
                        "Departure from Peebles & Xie's DDPM scheduler is "
                        "intentional — Eidex 2025 §4 uses RFlow for the "
                        "DiT-3D baseline, and this keeps the only axis "
                        "isolated against VENA as the backbone."
                    ),
                    "loss": (
                        "L1 on velocity u_t = z_T1c - z_noise (Eidex 2025 "
                        "Eq. 4) — not L2 on epsilon (Peebles & Xie)."
                    ),
                    "conditioning": (
                        "channel-wise concat — [noisy_z_T1c, z_T1pre, "
                        "z_FLAIR]; in_channels = latent_channels * 3 = 12. "
                        "y=None (no class label) at every forward call."
                    ),
                    "augmentation": (
                        "none — dataset reads cohort.latent_h5 only; "
                        "deterministic by contract (pinned by "
                        "test_dataset_is_deterministic)."
                    ),
                    "vae_handling": (
                        "vena_maisi_v2_symmetric_baseline — same MAISI-V2 "
                        "latents both for VENA and this competitor; the VAE "
                        "is symmetric across the comparison so the VAE-choice "
                        "confound does not contaminate VENA-vs-3D-DiT deltas."
                    ),
                    "intensity_norm_for_metrics": (
                        "VENA percentile_normalise(99.5, foreground_only) "
                        "for the real-T1c reference at inference time. "
                        "Decoded predictions live in the MAISI [0, 1] space "
                        "natively."
                    ),
                    "ema": "none (paper and wrapper)",
                    "amp": (
                        "enabled by default — same posture as T1C-RFlow "
                        "(upstream uses GradScaler + autocast(fp16); the "
                        "Peebles & Xie DiT paper is silent on AMP for 3D)."
                    ),
                    "t2_flair_substitute": (
                        "FLAIR only — UCSF-PDGM has T2 and FLAIR separately; "
                        "Eidex 2025's 'T2-FLAIR' is BraTS-combined. We use "
                        "FLAIR as the closest 1:1 substitute, same as "
                        "T1C-RFlow."
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
        # Vendored upstream must be present (the runner imports DiT3DWrapper
        # from it at runtime).
        upstream_dir = self.repo_root / "src" / "external" / "dit_3d" / "upstream"
        if not (upstream_dir / "dit3d_wrapper.py").is_file():
            raise FileNotFoundError(
                f"vendored DiT3D upstream missing at {upstream_dir}; "
                "re-vendor per src/external/dit_3d/UPSTREAM.md"
            )
        anchor = self.cfg.runtime.experiments_root.parent.parent
        if not anchor.exists():
            raise FileNotFoundError(
                f"experiments_root grandparent does not exist: {anchor}"
            )
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

        from vena.competitors.dit_3d import train_dit_3d

        train_dit_3d(self._build_runner_cfg(), run_dir)

        self._write_decision(run_dir, completed=True)
        return run_dir

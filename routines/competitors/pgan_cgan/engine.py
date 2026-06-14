"""Engine for the pGAN-cGAN competitor benchmark routine.

Wires a Pydantic config to ``vena.competitors.pgan_cgan.train_pgan``, generates a
deterministic run id, writes ``decision.json`` (competitor schema 1.0), and
returns the run directory path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from types import SimpleNamespace

logger = logging.getLogger(__name__)


COMPETITOR_NAME = "pgan"
COMPETITOR_PAPER = "Dar et al. 2019, IEEE TMI 38(10):2375–2388"
COMPETITOR_DOI = "10.1109/TMI.2019.2901750"
COMPETITOR_REPO = "https://github.com/icon-lab/pGAN-cGAN"
PRODUCER_VERSION = "0.1.0"


def _read_upstream_sha() -> str:
    """Return the vendored upstream SHA from ``src/external/pgan_cgan/UPSTREAM_SHA.txt``."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    sha_file = repo_root / "src" / "external" / "pgan_cgan" / "UPSTREAM_SHA.txt"
    if sha_file.is_file():
        return sha_file.read_text().strip()
    return "unknown"


def _short_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------
class DataCfg(BaseModel):
    """Where to read VENA's image-domain H5 from and how to slice it."""
    image_h5: Path
    fold: int = 0
    input_modalities: tuple[str, ...] = ("t1pre", "t2", "flair")
    target_modality: str = "t1c"
    image_size: int = 256
    min_brain_voxels: int = 1000
    max_train_patients: int | None = None


class HyperParamsCfg(BaseModel):
    """pGAN training hyperparameters. Defaults match the paper's IXI demo recipe."""
    input_nc: int = 3
    output_nc: int = 1
    ngf: int = 64
    ndf: int = 64
    n_layers_D: int = 3
    norm: str = "instance"
    init_type: str = "normal"
    no_dropout: bool = False
    pool_size: int = 0
    lambda_A: float = 100.0
    lambda_vgg: float = 100.0
    lambda_adv: float = 1.0
    lr: float = 2.0e-4
    beta1: float = 0.5
    lr_policy: Literal["lambda", "step", "plateau"] = "lambda"
    lr_decay_iters: int = 50
    epoch_count: int = 1
    niter: int = 50
    niter_decay: int = 50
    batchSize: int = 1
    save_epoch_freq: int = 1
    log_every: int = 25
    num_workers: int = 4


class RuntimeCfg(BaseModel):
    """Where to put artifacts and which platform we are on."""
    experiments_root: Path
    platform: Literal["server3", "loginexa", "picasso", "local"]
    tag: str
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    seed: int = 1337


class PGANCompetitorConfig(BaseModel):
    """Top-level config for the pGAN competitor routine."""
    runtime: RuntimeCfg
    data: DataCfg
    hp: HyperParamsCfg

    @model_validator(mode="after")
    def _check_channels(self) -> "PGANCompetitorConfig":
        if self.hp.input_nc != len(self.data.input_modalities):
            raise ValueError(
                f"hp.input_nc={self.hp.input_nc} but len(data.input_modalities)="
                f"{len(self.data.input_modalities)}"
            )
        if self.hp.output_nc != 1:
            raise ValueError(
                f"pGAN target is single-channel; hp.output_nc={self.hp.output_nc}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "PGANCompetitorConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class PGANCompetitorEngine:
    """Thin orchestrator — Pydantic cfg → run_dir → train_pgan → decision.json."""

    def __init__(self, cfg: PGANCompetitorConfig, config_yaml_path: Path) -> None:
        self.cfg = cfg
        self.config_yaml_path = Path(config_yaml_path).resolve()
        self.repo_root = Path(__file__).resolve().parents[3]

    def _generate_run_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        sha = _short_git_sha(self.repo_root)
        return f"{ts}_competitor_pgan_{self.cfg.runtime.tag}_{sha}"

    def _build_runner_cfg(self) -> SimpleNamespace:
        """Translate Pydantic config → plain SimpleNamespace for ``train_pgan``."""
        d, h = self.cfg.data, self.cfg.hp
        return SimpleNamespace(
            image_h5=d.image_h5,
            fold=d.fold,
            input_modalities=d.input_modalities,
            target_modality=d.target_modality,
            image_size=d.image_size,
            min_brain_voxels=d.min_brain_voxels,
            max_train_patients=d.max_train_patients,
            input_nc=h.input_nc,
            output_nc=h.output_nc,
            ngf=h.ngf,
            ndf=h.ndf,
            n_layers_D=h.n_layers_D,
            norm=h.norm,
            init_type=h.init_type,
            no_dropout=h.no_dropout,
            pool_size=h.pool_size,
            lambda_A=h.lambda_A,
            lambda_vgg=h.lambda_vgg,
            lambda_adv=h.lambda_adv,
            lr=h.lr,
            beta1=h.beta1,
            lr_policy=h.lr_policy,
            lr_decay_iters=h.lr_decay_iters,
            epoch_count=h.epoch_count,
            niter=h.niter,
            niter_decay=h.niter_decay,
            batchSize=h.batchSize,
            save_epoch_freq=h.save_epoch_freq,
            log_every=h.log_every,
            num_workers=h.num_workers,
            gpu_ids=self.cfg.runtime.gpu_ids,
        )

    def _file_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_decision(self, run_dir: Path, completed: bool) -> None:
        h5 = self.cfg.data.image_h5
        decision = {
            "schema_version": "1.0",
            "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "producer": f"routines.competitors.pgan_cgan:{PRODUCER_VERSION}",
            "completed": completed,
            "competitor": {
                "name": COMPETITOR_NAME,
                "paper": COMPETITOR_PAPER,
                "doi": COMPETITOR_DOI,
                "upstream_repo": COMPETITOR_REPO,
                "upstream_sha": _read_upstream_sha(),
            },
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "tag": self.cfg.runtime.tag,
            "platform": self.cfg.runtime.platform,
            "seed": self.cfg.runtime.seed,
            "git_sha": _short_git_sha(self.repo_root),
            "data": {
                "image_h5": str(h5),
                "image_h5_size_bytes": h5.stat().st_size if h5.is_file() else None,
                "fold": self.cfg.data.fold,
                "input_modalities": list(self.cfg.data.input_modalities),
                "target_modality": self.cfg.data.target_modality,
                "image_size": self.cfg.data.image_size,
                "min_brain_voxels": self.cfg.data.min_brain_voxels,
                "max_train_patients": self.cfg.data.max_train_patients,
            },
            "hyperparams": self.cfg.hp.model_dump(),
            "runtime": {
                "gpu_ids": self.cfg.runtime.gpu_ids,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_home": os.environ.get("TORCH_HOME"),
            },
        }
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    def _preflight(self) -> None:
        h5 = self.cfg.data.image_h5
        if not h5.is_file():
            raise FileNotFoundError(f"image H5 missing: {h5}")
        # Walk up two levels (bucket + per-competitor subdir) and ensure that
        # ancestor exists — this catches typos in experiments_root while still
        # letting us create the competitor's own bucket dir on first run.
        anchor = self.cfg.runtime.experiments_root.parent.parent
        if not anchor.exists():
            raise FileNotFoundError(
                f"experiments_root grandparent does not exist: {anchor}"
            )
        self.cfg.runtime.experiments_root.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        self._preflight()

        # Reproducibility — seed torch/numpy/python only here; the dataset is
        # deterministic by construction, so this only affects the discriminator
        # / VGG perceptual loss init RNG.
        import random as _random
        import numpy as _np
        import torch as _torch
        seed = self.cfg.runtime.seed
        _random.seed(seed); _np.random.seed(seed); _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)

        run_id = self._generate_run_id()
        run_dir = self.cfg.runtime.experiments_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Persist the resolved config.
        shutil.copy2(self.config_yaml_path, run_dir / "config.original.yaml")
        (run_dir / "config.resolved.json").write_text(
            json.dumps(self.cfg.model_dump(mode="json"), indent=2, default=str)
        )
        # Write a preliminary decision.json so a partial / crashed run is still tracked.
        self._write_decision(run_dir, completed=False)
        logger.info("run_dir = %s", run_dir)

        from vena.competitors.pgan_cgan import train_pgan

        train_pgan(self._build_runner_cfg(), run_dir)

        # Final decision.json now that training reached the end.
        self._write_decision(run_dir, completed=True)
        return run_dir

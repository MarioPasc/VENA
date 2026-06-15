"""Engine for the SynDiff competitor benchmark routine.

Wires a Pydantic config to :func:`vena.competitors.syndiff.train_syndiff`,
generates a deterministic run id, writes ``decision.json`` (competitor schema
1.0 with a ``competitor.deviations`` extension block), and returns the run
directory path.

Citation
--------
Özbey *et al.* 2023, "Unsupervised Medical Image Translation with Adversarial
Diffusion Models," IEEE Transactions on Medical Imaging, arXiv:2207.08208v3.
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
from types import SimpleNamespace
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


COMPETITOR_NAME = "syndiff"
COMPETITOR_PAPER = "Özbey et al. 2023, IEEE TMI, arXiv:2207.08208v3"
COMPETITOR_DOI = "arXiv:2207.08208v3"
COMPETITOR_REPO = "https://github.com/icon-lab/SynDiff"
PRODUCER_VERSION = "0.1.0"


def _read_upstream_sha() -> str:
    """Return the vendored upstream SHA from ``src/external/syndiff/UPSTREAM_SHA.txt``."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    sha_file = repo_root / "src" / "external" / "syndiff" / "UPSTREAM_SHA.txt"
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
    """Where to read VENA's image-domain data from.

    Two modes (exactly one must be set):

    1. **Single-cohort** (smoke / sanity): ``image_h5`` points at one VENA
       cohort image H5.
    2. **Multi-cohort** (production / fair comparison vs VENA):
       ``corpus_registry`` points at a VENA corpus-registry JSON; the runner
       concatenates every cohort with ``role="cv"``.
    """

    image_h5: Path | None = None
    corpus_registry: Path | None = None
    fold: int = 0
    source_modality: Literal["t1pre", "t2", "flair"] = "t1pre"
    target_modality: Literal["t1c"] = "t1c"
    max_patients_per_cohort: int | None = None
    max_train_patients: int | None = None  # single-cohort smoke knob
    min_brain_voxels: int = 1000
    image_size: int = 256
    cohort_path_overrides: dict[str, Path] = Field(default_factory=dict)


class HyperParamsCfg(BaseModel):
    """SynDiff hyperparameters. Defaults follow the README invocation
    (Özbey et al., IEEE TMI 2023). The README differs from the paper text on
    epoch count — see ``UPSTREAM.md`` for the divergence table.
    """

    # NCSN++ trunk
    num_channels_dae: int = 64
    ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.0
    embedding_type: Literal["positional", "fourier"] = "positional"
    nz: int = 100
    z_emb_dim: int = 256
    t_emb_dim: int = 256
    n_mlp: int = 3

    # Discriminator
    ngf: int = 64

    # Diffusion schedule
    num_timesteps: int = 4               # T/k from paper (T=1000, k=250)
    beta_min: float = 0.1
    beta_max: float = 20.0

    # Optimisation
    lr_g: float = 1.6e-4
    lr_d: float = 1.0e-4
    beta1: float = 0.5
    beta2: float = 0.9
    r1_gamma: float = 1.0
    lazy_reg: int = 10
    lambda_l1_loss: float = 0.5
    use_ema: bool = True
    ema_decay: float = 0.999
    no_lr_decay: bool = False

    # Training schedule + early stop
    max_epochs: int = 50
    patience: int = 0  # 0 = no early-stop (50-epoch budget is short anyway)
    save_epoch_freq: int = 10
    log_every: int = 100
    batch_size: int = 1
    num_workers: int = 0


class RuntimeCfg(BaseModel):
    """Where to put artifacts and which platform we are on."""

    experiments_root: Path
    platform: Literal["server3", "loginexa", "picasso", "local"]
    tag: str
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    seed: int = 1337


class SynDiffCompetitorConfig(BaseModel):
    """Top-level config for the SynDiff competitor routine."""

    runtime: RuntimeCfg
    data: DataCfg
    hp: HyperParamsCfg

    @model_validator(mode="after")
    def _check_consistency(self) -> "SynDiffCompetitorConfig":
        if bool(self.data.image_h5) == bool(self.data.corpus_registry):
            raise ValueError(
                "DataCfg requires exactly one of {image_h5, corpus_registry} to be set."
            )
        if self.data.image_size % 32 != 0:
            raise ValueError(
                f"image_size must be divisible by 32 (6-level NCSN++); "
                f"got {self.data.image_size}"
            )
        if self.data.source_modality == self.data.target_modality:
            raise ValueError(
                f"source_modality and target_modality must differ "
                f"(both were {self.data.source_modality!r})"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "SynDiffCompetitorConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SynDiffCompetitorEngine:
    """Thin orchestrator — Pydantic cfg → run_dir → train_syndiff → decision.json."""

    def __init__(
        self,
        cfg: SynDiffCompetitorConfig,
        config_yaml_path: Path,
    ) -> None:
        self.cfg = cfg
        self.config_yaml_path = Path(config_yaml_path).resolve()
        self.repo_root = Path(__file__).resolve().parents[3]

    def _generate_run_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        sha = _short_git_sha(self.repo_root)
        return f"{ts}_competitor_syndiff_{self.cfg.runtime.tag}_{sha}"

    def _build_runner_cfg(self) -> SimpleNamespace:
        """Translate Pydantic config → SimpleNamespace for ``train_syndiff``."""
        d, h, r = self.cfg.data, self.cfg.hp, self.cfg.runtime
        return SimpleNamespace(
            # Data
            corpus_registry=d.corpus_registry,
            image_h5=d.image_h5,
            cohort_path_overrides=d.cohort_path_overrides,
            max_patients_per_cohort=d.max_patients_per_cohort,
            max_train_patients=d.max_train_patients,
            fold=d.fold,
            source_modality=d.source_modality,
            target_modality=d.target_modality,
            min_brain_voxels=d.min_brain_voxels,
            image_size=d.image_size,
            # Hyperparameters
            num_channels_dae=h.num_channels_dae,
            ch_mult=h.ch_mult,
            num_res_blocks=h.num_res_blocks,
            attn_resolutions=h.attn_resolutions,
            dropout=h.dropout,
            embedding_type=h.embedding_type,
            nz=h.nz,
            z_emb_dim=h.z_emb_dim,
            t_emb_dim=h.t_emb_dim,
            n_mlp=h.n_mlp,
            ngf=h.ngf,
            num_timesteps=h.num_timesteps,
            beta_min=h.beta_min,
            beta_max=h.beta_max,
            lr_g=h.lr_g,
            lr_d=h.lr_d,
            beta1=h.beta1,
            beta2=h.beta2,
            r1_gamma=h.r1_gamma,
            lazy_reg=h.lazy_reg,
            lambda_l1_loss=h.lambda_l1_loss,
            use_ema=h.use_ema,
            ema_decay=h.ema_decay,
            no_lr_decay=h.no_lr_decay,
            max_epochs=h.max_epochs,
            patience=h.patience,
            save_epoch_freq=h.save_epoch_freq,
            log_every=h.log_every,
            batch_size=h.batch_size,
            num_workers=h.num_workers,
            # Runtime
            gpu_ids=r.gpu_ids,
            seed=r.seed,
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
            "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "producer": f"routines.competitors.syndiff:{PRODUCER_VERSION}",
            "completed": completed,
            "competitor": {
                "name": COMPETITOR_NAME,
                "paper": COMPETITOR_PAPER,
                "doi": COMPETITOR_DOI,
                "upstream_repo": COMPETITOR_REPO,
                "upstream_sha": _read_upstream_sha(),
                # Per-file licence carve-outs — flagged for any future
                # release decision; see src/external/syndiff/UPSTREAM.md.
                "license_caveat": (
                    "Top-level MIT, but backbones/ncsnpp_generator_adagn.py, "
                    "backbones/discriminator.py, and utils/EMA.py carry the "
                    "NVIDIA Source Code License (non-commercial research "
                    "only). The top-level MIT does not override these. "
                    "Acceptable for the VENA research benchmark; redistribution "
                    "in any commercial form requires backbone replacement or "
                    "explicit NVIDIA grant."
                ),
                "deviations": {
                    "dimensionality": (
                        "2D per-axial-slice — paper is 2D; VENA's other "
                        "competitors and our own model are 3D. Per-slice "
                        "predictions are reassembled to 3D volumes for "
                        "metric parity downstream."
                    ),
                    "modality_pair": (
                        "one-to-one source→target — SynDiff is bilateral, "
                        "training two diffusive + two non-diffusive generators "
                        "jointly via cycle-consistency. The 'source' is the "
                        "configured source_modality; 'target' is fixed at t1c. "
                        "Per-source panel matches the pgan_cgan pattern."
                    ),
                    "ema": (
                        "enabled on the four generators (use_ema=true, "
                        "ema_decay=0.999) — matches the README example. "
                        "Best/latest/periodic checkpoints save the EMA "
                        "shadow weights, not the raw step weights."
                    ),
                    "training_loop": (
                        "single-GPU only — DistributedSampler / "
                        "broadcast_params / DistributedDataParallel "
                        "from upstream train.py are stripped. README "
                        "example uses --num_process_per_node 1; multi-GPU "
                        "adds no value at our budget."
                    ),
                    "set_detect_anomaly": (
                        "removed — train.py:582 hard-codes "
                        "torch.autograd.set_detect_anomaly(True) per iteration "
                        "(upstream issue #43). Our runner does not replicate "
                        "this; PATCHES.md P2 documents the in-place removal "
                        "for the vendored reference copy."
                    ),
                    "epoch_count": (
                        "paper §IV.B states 50 epochs; README invocation says "
                        "500. We follow paper (50), matching VENA policy "
                        "2026-06-15 of paper-text over unreviewed README."
                    ),
                    "intensity_norm_for_metrics": (
                        "VENA percentile_normalise(99.5, foreground_only=True) "
                        "applied per-patient to both source and target. Matches "
                        "VENA's encoding convention; downstream metrics compare "
                        "decoded predictions against percentile-normalised "
                        "targets, not raw intensities."
                    ),
                    "augmentation": (
                        "none — dataset is deterministic (pinned by "
                        "test_dataset_is_deterministic). VENA owns the "
                        "augmentation regime; competitor loaders never apply "
                        "augmentation."
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
                "image_h5": str(self.cfg.data.image_h5)
                if self.cfg.data.image_h5 else None,
                "corpus_registry": str(self.cfg.data.corpus_registry)
                if self.cfg.data.corpus_registry else None,
                "cohort_path_overrides": {
                    k: str(v) for k, v in self.cfg.data.cohort_path_overrides.items()
                },
                "fold": self.cfg.data.fold,
                "source_modality": self.cfg.data.source_modality,
                "target_modality": self.cfg.data.target_modality,
                "max_patients_per_cohort": self.cfg.data.max_patients_per_cohort,
                "max_train_patients": self.cfg.data.max_train_patients,
                "image_size": self.cfg.data.image_size,
                "min_brain_voxels": self.cfg.data.min_brain_voxels,
            },
            "hyperparams": json.loads(self.cfg.hp.model_dump_json()),
            "runtime": {
                "gpu_ids": self.cfg.runtime.gpu_ids,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    def _preflight(self) -> None:
        if self.cfg.data.image_h5 is not None and not self.cfg.data.image_h5.is_file():
            raise FileNotFoundError(
                f"image_h5 missing: {self.cfg.data.image_h5}"
            )
        if self.cfg.data.corpus_registry is not None and not self.cfg.data.corpus_registry.is_file():
            raise FileNotFoundError(
                f"corpus_registry missing: {self.cfg.data.corpus_registry}"
            )
        anchor = self.cfg.runtime.experiments_root.parent.parent
        if not anchor.exists():
            raise FileNotFoundError(
                f"experiments_root grandparent does not exist: {anchor}"
            )
        self.cfg.runtime.experiments_root.mkdir(parents=True, exist_ok=True)
        upstream_dir = (
            self.repo_root / "src" / "external" / "syndiff" / "upstream"
        )
        if not upstream_dir.is_dir():
            raise FileNotFoundError(
                f"vendored SynDiff upstream missing: {upstream_dir}"
            )

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
        # Preliminary decision.json so partial / crashed runs are still tracked.
        self._write_decision(run_dir, completed=False)
        logger.info("run_dir = %s", run_dir)

        from vena.competitors.syndiff import train_syndiff

        train_syndiff(self._build_runner_cfg(), run_dir)

        self._write_decision(run_dir, completed=True)
        return run_dir

"""Engine for the ResViT competitor benchmark routine.

Wires a Pydantic config to ``vena.competitors.resvit.train_resvit``, generates a
deterministic run id, writes ``decision.json`` (competitor schema 1.0), and
returns the run directory path. The engine drives both stages of ResViT's
two-stage curriculum in a single ``run()`` call — see
``vena.competitors.resvit.runner`` for the stage transitions.
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


COMPETITOR_NAME = "resvit"
COMPETITOR_PAPER = "Dalmaz, Yurt, Çukur 2022, IEEE TMI 41(10):2598–2614"
COMPETITOR_DOI = "10.1109/TMI.2022.3167808"
COMPETITOR_REPO = "https://github.com/icon-lab/ResViT"
PRODUCER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    """Resolve VENA repo root from this file (engine.py)."""
    return Path(__file__).resolve().parents[3]


def _read_upstream_sha() -> str:
    """Return the vendored upstream SHA from ``src/external/resvit/UPSTREAM_SHA.txt``."""
    sha_file = _repo_root() / "src" / "external" / "resvit" / "UPSTREAM_SHA.txt"
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


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_vit_npz() -> Path:
    """Default location of the vendored R50+ViT-B_16.npz."""
    return (_repo_root() / "src" / "external" / "resvit" / "upstream"
            / "checkpoints" / "R50+ViT-B_16.npz")


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------
class DataCfg(BaseModel):
    """Where to read VENA's image-domain data from and how to slice it.

    Two modes:

    1. **Single-cohort** (smoke / sanity): set ``image_h5`` to a UCSF-PDGM-schema
       H5; the runner builds a ``UCSFPDGMSliceDataset`` directly. Used for fast
       per-cohort sanity checks.
    2. **Multi-cohort** (production / fair comparison vs VENA): set
       ``corpus_registry`` to a VENA corpus-registry JSON; the runner builds a
       ``MultiCohortImageSliceDataset`` that concatenates every cohort with
       ``role="cv"``. This matches VENA's FM-train data path exactly.

    Exactly one of ``image_h5`` or ``corpus_registry`` must be set.
    """

    image_h5: Path | None = None
    corpus_registry: Path | None = None
    fold: int = 0
    input_modalities: tuple[str, ...] = ("t1pre", "t2", "flair")
    target_modality: str = "t1c"
    image_size: int = 256
    min_brain_voxels: int = 1000
    max_train_patients: int | None = None
    max_patients_per_cohort: int | None = None
    cohort_path_overrides: dict[str, Path] = Field(default_factory=dict)


class HyperParamsCfg(BaseModel):
    """ResViT training hyperparameters. Defaults follow the paper's two-stage recipe."""

    # I/O channels
    input_nc: int = 3
    output_nc: int = 1

    # Architecture knobs (shared by both stages)
    ngf: int = 64
    ndf: int = 64
    n_layers_D: int = 3
    norm: str = "instance"
    init_type: str = "normal"
    no_dropout: bool = False
    vit_name: Literal["Res-ViT-B_16", "Res-ViT-L_16"] = "Res-ViT-B_16"

    # Upstream model class (must take variable input_nc — see UPSTREAM.md table).
    upstream_model: Literal["resvit_one"] = "resvit_one"

    # GAN
    pool_size: int = 0

    # Loss weights (paper: λ_adv=1, λ_A=100)
    lambda_A: float = 100.0
    lambda_adv: float = 1.0

    # Stage 1 — CNN pretrain (lr 2e-4, 50+50 epochs in the paper)
    pretrain_niter: int = 50
    pretrain_niter_decay: int = 50
    pretrain_lr: float = 2.0e-4

    # Stage 2 — ART fine-tune (lr 1e-3, 25+25 epochs in the paper)
    niter: int = 25
    niter_decay: int = 25
    lr: float = 1.0e-3

    beta1: float = 0.5
    lr_decay_iters: int = 50

    # Loop knobs
    batchSize: int = 1
    log_every: int = 25
    num_workers: int = 4
    # Patience-based early stopping (stage 2 only). 0 disables.
    patience: int = 0

    # Paper-budget caps — when set, the per-stage training loop breaks as soon
    # as the cumulative slice count reaches this value, regardless of the
    # configured ``niter + niter_decay`` epochs. This is the mechanism used to
    # match ResViT's paper data-exposure budget (Dalmaz et al. 2022 §III.B:
    # stage 1 sees ~250 000 slices, stage 2 sees ~125 000 slices on BRATS
    # many-to-one) on VENA's much larger multi-cohort corpus (~287 000
    # slices/epoch). When ``None``, training runs the full epoch budget.
    pretrain_max_slices: int | None = None
    max_slices: int | None = None


class RuntimeCfg(BaseModel):
    """Where to put artifacts and which platform we are on."""

    experiments_root: Path
    platform: Literal["server3", "loginexa", "picasso", "local"]
    tag: str
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    seed: int = 1337
    # Absolute path to R50+ViT-B_16.npz. Defaults to the vendored copy.
    vit_init_npz: Path | None = None


class ResViTCompetitorConfig(BaseModel):
    """Top-level config for the ResViT competitor routine."""

    runtime: RuntimeCfg
    data: DataCfg
    hp: HyperParamsCfg

    @model_validator(mode="after")
    def _check_axes(self) -> "ResViTCompetitorConfig":
        if self.hp.input_nc != len(self.data.input_modalities):
            raise ValueError(
                f"hp.input_nc={self.hp.input_nc} but len(data.input_modalities)="
                f"{len(self.data.input_modalities)}"
            )
        if self.hp.output_nc != 1:
            raise ValueError(
                f"ResViT target is single-channel; hp.output_nc={self.hp.output_nc}"
            )
        if self.data.image_size % 16 != 0:
            raise ValueError(
                f"data.image_size must be divisible by 16 (ResViT CNN ×4 + ViT 16×16 patch); "
                f"got {self.data.image_size}"
            )
        if bool(self.data.image_h5) == bool(self.data.corpus_registry):
            raise ValueError(
                "DataCfg requires exactly one of {image_h5, corpus_registry} to be set."
            )
        if self.hp.pretrain_niter < 1 or self.hp.niter < 1:
            raise ValueError(
                "Both stages must run at least 1 epoch (pretrain_niter, niter ≥ 1)."
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "ResViTCompetitorConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ResViTCompetitorEngine:
    """Thin orchestrator — Pydantic cfg → run_dir → train_resvit → decision.json."""

    def __init__(self, cfg: ResViTCompetitorConfig, config_yaml_path: Path) -> None:
        self.cfg = cfg
        self.config_yaml_path = Path(config_yaml_path).resolve()
        self.repo_root = _repo_root()
        # Resolve absolute path to ViT init checkpoint; fall back to vendored copy.
        self.vit_init_npz = (
            Path(cfg.runtime.vit_init_npz).resolve()
            if cfg.runtime.vit_init_npz is not None
            else _default_vit_npz()
        )

    def _generate_run_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        sha = _short_git_sha(self.repo_root)
        return f"{ts}_competitor_resvit_{self.cfg.runtime.tag}_{sha}"

    def _build_runner_cfg(self) -> SimpleNamespace:
        """Translate Pydantic config → plain SimpleNamespace for ``train_resvit``."""
        d, h = self.cfg.data, self.cfg.hp
        return SimpleNamespace(
            # Data
            image_h5=d.image_h5,
            corpus_registry=d.corpus_registry,
            cohort_path_overrides=d.cohort_path_overrides,
            max_patients_per_cohort=d.max_patients_per_cohort,
            fold=d.fold,
            input_modalities=d.input_modalities,
            target_modality=d.target_modality,
            image_size=d.image_size,
            min_brain_voxels=d.min_brain_voxels,
            max_train_patients=d.max_train_patients,
            # Architecture
            input_nc=h.input_nc,
            output_nc=h.output_nc,
            ngf=h.ngf,
            ndf=h.ndf,
            n_layers_D=h.n_layers_D,
            norm=h.norm,
            init_type=h.init_type,
            no_dropout=h.no_dropout,
            vit_name=h.vit_name,
            upstream_model=h.upstream_model,
            # GAN
            pool_size=h.pool_size,
            # Losses
            lambda_A=h.lambda_A,
            lambda_adv=h.lambda_adv,
            # Stage 1 / stage 2 schedules
            pretrain_niter=h.pretrain_niter,
            pretrain_niter_decay=h.pretrain_niter_decay,
            pretrain_lr=h.pretrain_lr,
            niter=h.niter,
            niter_decay=h.niter_decay,
            lr=h.lr,
            beta1=h.beta1,
            lr_decay_iters=h.lr_decay_iters,
            # Loop knobs
            batchSize=h.batchSize,
            log_every=h.log_every,
            num_workers=h.num_workers,
            patience=h.patience,
            pretrain_max_slices=h.pretrain_max_slices,
            max_slices=h.max_slices,
            # ViT init
            vit_init_npz=self.vit_init_npz,
            # GPU
            gpu_ids=self.cfg.runtime.gpu_ids,
        )

    def _write_decision(self, run_dir: Path, completed: bool) -> None:
        vit_sha = _file_sha256(self.vit_init_npz) if self.vit_init_npz.is_file() else None
        decision = {
            "schema_version": "1.0",
            "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "producer": f"routines.competitors.resvit:{PRODUCER_VERSION}",
            "completed": completed,
            "competitor": {
                "name": COMPETITOR_NAME,
                "paper": COMPETITOR_PAPER,
                "doi": COMPETITOR_DOI,
                "upstream_repo": COMPETITOR_REPO,
                "upstream_sha": _read_upstream_sha(),
                "upstream_model": self.cfg.hp.upstream_model,
                "vit_init_npz": str(self.vit_init_npz),
                "vit_init_npz_sha256": vit_sha,
                "deviations": {
                    "input_modalities":
                        f"{self.cfg.hp.input_nc}-channel "
                        f"{list(self.cfg.data.input_modalities)}; "
                        "paper demonstrates 2-channel many-to-one. SWAN excluded.",
                    "model_class":
                        "resvit_one (channel-parametric) used instead of resvit_many "
                        "(hardcoded 2-channel) — see UPSTREAM.md paper-vs-code table.",
                    "no_augmentation":
                        "VENA owns augmentation; competitor wrapper is deterministic.",
                    "datasets_in_paper":
                        "IXI / BRATS / pelvic MRI-CT — VENA evaluates on UCSF-PDGM + "
                        "multi-cohort glioma union.",
                    "ema_or_visdom":
                        "Visdom logging bypassed; metrics written to CSV directly.",
                    "slice_budget_cap": (
                        f"Paper stage 1 sees 100 ep × 2500 slices = 250 000 slices; "
                        f"stage 2 sees 50 ep × 2500 = 125 000 slices. VENA's "
                        f"multi-cohort fold-0 train union has 287 117 slices/epoch, so "
                        f"running 150 epochs verbatim would over-budget ResViT by ~114×. "
                        f"Engine caps per-stage slice exposure via "
                        f"pretrain_max_slices={self.cfg.hp.pretrain_max_slices} and "
                        f"max_slices={self.cfg.hp.max_slices} (None=no cap). "
                        f"LR schedule stays at paper-recipe niter+niter_decay but "
                        f"the cap exits before the decay phase triggers."
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
                "image_h5": str(self.cfg.data.image_h5) if self.cfg.data.image_h5 else None,
                "corpus_registry": (str(self.cfg.data.corpus_registry)
                                    if self.cfg.data.corpus_registry else None),
                "cohort_path_overrides": {k: str(v) for k, v
                                          in self.cfg.data.cohort_path_overrides.items()},
                "fold": self.cfg.data.fold,
                "input_modalities": list(self.cfg.data.input_modalities),
                "target_modality": self.cfg.data.target_modality,
                "image_size": self.cfg.data.image_size,
                "min_brain_voxels": self.cfg.data.min_brain_voxels,
                "max_train_patients": self.cfg.data.max_train_patients,
                "max_patients_per_cohort": self.cfg.data.max_patients_per_cohort,
            },
            "hyperparams": self.cfg.hp.model_dump(),
            "runtime": {
                "gpu_ids": self.cfg.runtime.gpu_ids,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_home": os.environ.get("TORCH_HOME"),
                "vit_init_npz": str(self.vit_init_npz),
            },
        }
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    def _preflight(self) -> None:
        if self.cfg.data.image_h5 is not None and not self.cfg.data.image_h5.is_file():
            raise FileNotFoundError(f"image H5 missing: {self.cfg.data.image_h5}")
        if (self.cfg.data.corpus_registry is not None
                and not self.cfg.data.corpus_registry.is_file()):
            raise FileNotFoundError(
                f"corpus_registry missing: {self.cfg.data.corpus_registry}"
            )
        if not self.vit_init_npz.is_file():
            raise FileNotFoundError(
                f"ViT init checkpoint missing: {self.vit_init_npz}\n"
                "Download with:\n"
                f"  curl -sSL -o {self.vit_init_npz} "
                "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"
            )
        # Walk up two levels (bucket + per-competitor subdir) and ensure that
        # ancestor exists — this catches typos in experiments_root.
        anchor = self.cfg.runtime.experiments_root.parent.parent
        if not anchor.exists():
            raise FileNotFoundError(
                f"experiments_root grandparent does not exist: {anchor}"
            )
        self.cfg.runtime.experiments_root.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        self._preflight()

        # Reproducibility — seed torch/numpy/python only here; the dataset is
        # deterministic by construction.
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

        shutil.copy2(self.config_yaml_path, run_dir / "config.original.yaml")
        (run_dir / "config.resolved.json").write_text(
            json.dumps(self.cfg.model_dump(mode="json"), indent=2, default=str)
        )
        # Write a preliminary decision.json so a partial / crashed run is still tracked.
        self._write_decision(run_dir, completed=False)
        logger.info("run_dir = %s", run_dir)

        from vena.competitors.resvit import train_resvit

        train_resvit(self._build_runner_cfg(), run_dir)

        # Final decision.json now that training reached the end.
        self._write_decision(run_dir, completed=True)
        return run_dir

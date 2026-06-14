"""Programmatic training entrypoint for the vendored pGAN model.

Imports the patched upstream model classes, builds an ``argparse.Namespace`` that
matches the upstream option schema, builds the VENA dataset, and drives the
training loop. Writes ``metrics/train_step.csv``, per-epoch checkpoints under
``checkpoints/``, and ``logs/train.log`` into ``run_dir``.

The Visdom-based ``util.visualizer.Visualizer`` is bypassed — we log to CSV and
to the file logger directly.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .dataset import UCSFPDGMSliceDataset


logger = logging.getLogger(__name__)


class PGANRunnerError(Exception):
    """Raised when the runner cannot proceed (missing VGG cache, bad cfg)."""


# ---------------------------------------------------------------------------
# Upstream import shim
# ---------------------------------------------------------------------------
# The vendored upstream uses relative imports (``from util import util``,
# ``from .base_model import BaseModel``). We add ``src/external/pgan_cgan/upstream``
# to sys.path *temporarily* so those resolve without polluting the global path.
_UPSTREAM_DIR = (Path(__file__).resolve().parent.parent.parent.parent
                 / "external" / "pgan_cgan" / "upstream")


def _import_pgan_model():
    """Import the vendored ``pGAN`` class with the upstream sys.path shim."""
    if not _UPSTREAM_DIR.is_dir():
        raise PGANRunnerError(f"vendored pGAN upstream not found at {_UPSTREAM_DIR}")
    sys_path_was = list(sys.path)
    sys.path.insert(0, str(_UPSTREAM_DIR))
    try:
        # Use a stable alias so multiple instantiations don't trigger import cache races.
        from models.pgan_model import pGAN  # type: ignore[import-not-found]
    finally:
        sys.path = sys_path_was
    return pGAN


# ---------------------------------------------------------------------------
# Configuration → upstream Namespace
# ---------------------------------------------------------------------------
def _build_opt(cfg, run_dir: Path) -> SimpleNamespace:
    """Translate VENA config → the SimpleNamespace pGAN's BaseModel expects.

    Every flag pGAN reads from its argparse must appear here. We do NOT call the
    upstream argparse parser — that would also create checkpoint dirs and an
    ``opt.txt`` we don't want.
    """
    checkpoints_dir = run_dir
    name = "checkpoints"
    (checkpoints_dir / name).mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        # Identification
        name=name,
        model="pGAN",
        dataset_mode="aligned_mat",
        which_direction="AtoB",
        # I/O channels
        input_nc=cfg.input_nc,
        output_nc=cfg.output_nc,
        # Architecture
        ngf=cfg.ngf,
        ndf=cfg.ndf,
        n_layers_D=cfg.n_layers_D,
        norm=cfg.norm,
        init_type=cfg.init_type,
        no_dropout=cfg.no_dropout,
        # GAN
        no_lsgan=False,
        pool_size=cfg.pool_size,
        # Loss weights
        lambda_A=cfg.lambda_A,
        lambda_vgg=cfg.lambda_vgg,
        lambda_adv=cfg.lambda_adv,
        # Optim
        lr=cfg.lr,
        beta1=cfg.beta1,
        # Schedule
        lr_policy=cfg.lr_policy,
        lr_decay_iters=cfg.lr_decay_iters,
        epoch_count=cfg.epoch_count,
        niter=cfg.niter,
        niter_decay=cfg.niter_decay,
        # I/O paths (BaseModel uses checkpoints_dir + name to build save_dir)
        checkpoints_dir=str(checkpoints_dir),
        # Continue/resume — VENA does not currently support resume here.
        continue_train=False,
        which_epoch="latest",
        isTrain=True,
        # GPU
        gpu_ids=cfg.gpu_ids,
        # Misc that pGAN reads but does not gate behaviour.
        batchSize=cfg.batchSize,
        serial_batches=False,
    )


# ---------------------------------------------------------------------------
# VGG cache check (Picasso compute nodes have no internet)
# ---------------------------------------------------------------------------
_VGG_WEIGHT_NAMES = (
    "vgg16-397923af.pth",
    "vgg16_features-amdegroot-88682ab5.pth",
)


def _verify_vgg_cache() -> Path:
    """Locate the cached VGG16 weights or raise a clear error."""
    candidates: list[Path] = []
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        candidates.append(Path(torch_home) / "hub" / "checkpoints")
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    for d in candidates:
        for name in _VGG_WEIGHT_NAMES:
            p = d / name
            if p.is_file():
                logger.info("Using cached VGG16 weights at %s", p)
                return p
    logger.warning(
        "VGG16 cache not found in %s — will try a live download. If you are on "
        "a Picasso compute node this will fail; pre-warm the cache on the login node:\n"
        "  TORCH_HOME=$HOME/.cache/torch python -c "
        "\"from torchvision.models import vgg16, VGG16_Weights; "
        "vgg16(weights=VGG16_Weights.DEFAULT)\"",
        [str(d) for d in candidates],
    )
    return Path("(none)")


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------
def train_pgan(cfg, run_dir: Path) -> Path:
    """Run pGAN training. Returns ``run_dir``."""
    from .dataset import UCSFPDGMSliceDataset

    run_dir = Path(run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # File handler captures the whole run independent of stdout redirection.
    fh = logging.FileHandler(run_dir / "logs" / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("pGAN runner — run_dir=%s", run_dir)
    _verify_vgg_cache()

    # GPU placement — pGAN's BaseModel uses torch.cuda.set_device on gpu_ids[0].
    if cfg.gpu_ids:
        torch.cuda.set_device(cfg.gpu_ids[0])

    opt = _build_opt(cfg, run_dir)

    pGAN = _import_pgan_model()
    model = pGAN()
    model.initialize(opt)
    logger.info("pGAN model initialised (input_nc=%d, output_nc=%d, ngf=%d)",
                opt.input_nc, opt.output_nc, opt.ngf)

    train_ds = UCSFPDGMSliceDataset(
        image_h5=cfg.image_h5,
        fold=cfg.fold,
        phase="train",
        input_modalities=cfg.input_modalities,
        target_modality=cfg.target_modality,
        image_size=cfg.image_size,
        min_brain_voxels=cfg.min_brain_voxels,
        max_patients=cfg.max_train_patients,
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batchSize,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=bool(cfg.gpu_ids),
        persistent_workers=cfg.num_workers > 0,
    )
    logger.info("Train loader: %d slices, batch=%d, workers=%d",
                len(train_ds), cfg.batchSize, cfg.num_workers)

    csv_path = run_dir / "metrics" / "train_step.csv"
    fields = ["epoch", "global_step", "iter_in_epoch",
              "G_GAN", "G_L1", "G_VGG", "D_real", "D_fake", "lr", "step_seconds"]
    with csv_path.open("w", newline="") as fh_csv:
        writer = csv.DictWriter(fh_csv, fieldnames=fields)
        writer.writeheader()

        epoch_csv = run_dir / "metrics" / "train_epoch.csv"
        with epoch_csv.open("w", newline="") as fh_epoch:
            epoch_writer = csv.DictWriter(
                fh_epoch,
                fieldnames=["epoch", "G_GAN_mean", "G_L1_mean", "G_VGG_mean",
                            "D_real_mean", "D_fake_mean", "wall_seconds"],
            )
            epoch_writer.writeheader()

            global_step = 0
            for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
                epoch_start = time.time()
                acc: dict[str, list[float]] = {
                    "G_GAN": [], "G_L1": [], "G_VGG": [], "D_real": [], "D_fake": [],
                }

                for it, batch in enumerate(loader):
                    t0 = time.time()
                    model.set_input(batch)
                    model.optimize_parameters()
                    errors = model.get_current_errors()
                    lr = model.optimizers[0].param_groups[0]["lr"]
                    dt = time.time() - t0

                    writer.writerow({
                        "epoch": epoch,
                        "global_step": global_step,
                        "iter_in_epoch": it,
                        "G_GAN": float(errors["G_GAN"]),
                        "G_L1": float(errors["G_L1"]),
                        "G_VGG": float(errors["G_VGG"]),
                        "D_real": float(errors["D_real"]),
                        "D_fake": float(errors["D_fake"]),
                        "lr": float(lr),
                        "step_seconds": dt,
                    })
                    fh_csv.flush()
                    for k in acc:
                        acc[k].append(float(errors[k]))
                    global_step += 1
                    if global_step % cfg.log_every == 0:
                        logger.info(
                            "epoch=%d step=%d G_L1=%.4f G_VGG=%.4f G_GAN=%.4f "
                            "D_real=%.4f D_fake=%.4f lr=%.2e dt=%.2fs",
                            epoch, global_step, errors["G_L1"], errors["G_VGG"],
                            errors["G_GAN"], errors["D_real"], errors["D_fake"], lr, dt,
                        )

                # End-of-epoch.
                wall = time.time() - epoch_start
                means = {f"{k}_mean": (sum(v) / len(v) if v else float("nan"))
                         for k, v in acc.items()}
                epoch_writer.writerow({"epoch": epoch, "wall_seconds": wall, **means})
                fh_epoch.flush()
                logger.info(
                    "epoch %d done in %.1fs — G_L1=%.4f G_VGG=%.4f G_GAN=%.4f",
                    epoch, wall, means["G_L1_mean"], means["G_VGG_mean"], means["G_GAN_mean"],
                )

                if epoch % cfg.save_epoch_freq == 0 or epoch == opt.niter + opt.niter_decay:
                    model.save(epoch)
                    logger.info("saved checkpoint %d_net_{G,D}.pth", epoch)
                model.save("latest")
                model.update_learning_rate()

    logger.info("pGAN training completed — %d epochs, %d steps", epoch, global_step)
    # Sentinel string consumed by skill completion checks.
    logger.info("pGAN-train completed")
    return run_dir

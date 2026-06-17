"""3D-Latent-Pix2Pix training loop (Isola *et al.* 2017 conditional-GAN recipe).

Generator: :class:`DiffusionModelUNetMaisi` instantiated at the paper-faithful
3-level config (same as :mod:`vena.competitors.t1c_rflow.runner`) and wrapped
in a :class:`_GeneratorUNetWrapper` that feeds **zero timesteps** so the
diffusion U-Net runs as a deterministic conditional generator. Discriminator:
:class:`_PatchDiscriminator3D` (4 strided ``Conv3d`` + InstanceNorm + LeakyReLU
layers, ``ndf=64``, terminal 1-channel patch-logits head). Loss: BCE
adversarial + ``lambda_l1`` × L1 with ``lambda_l1 = 100`` (Isola §3.2).

The architecture choices are identical to the vendored
``src/external/lpix2pix_3d/upstream/train_pix2pix_t1n_t2f.py`` on every
load-bearing axis (see ``UPSTREAM.md`` for the paper-vs-code table); the
short ``GeneratorUNetWrapper`` and ``PatchDiscriminator3D`` classes are
re-implemented here against MONAI primitives to avoid pulling the vendored
script's ``argparse``/``tqdm``/``matplotlib`` plumbing into VENA's import
graph.

What this runner adds on top of upstream
----------------------------------------
* Multi-cohort training over a VENA corpus registry (the upstream loader
  expects a single ``train/`` folder of ``.pt`` files).
* Per-step and per-epoch CSV logging
  (``metrics/train_step.csv`` with ``loss_g_total``, ``loss_g_adv``,
  ``loss_g_l1``, ``loss_d_real``, ``loss_d_fake``;
  ``metrics/train_epoch.csv`` with the per-loss means and ``wall_seconds``).
* ``best`` / ``latest`` / ``epoch_<N>`` checkpoints for **both** generator
  and discriminator with patience-based early stop on epoch-mean ``loss_g_l1``
  (the L1 component tracks reconstruction quality more directly than the
  total — same selection metric Isola *et al.* 2017 §6 reports).
* Self-contained log file (``logs/train.log``) attached as a ``FileHandler``
  at engine entry.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import (
    MultiCohortPix2PixLatentDataset,
    Pix2PixLatentDataset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class Pix2PixRunnerError(Exception):
    """Raised on misconfiguration or training-time failure."""


@dataclass
class _RunnerArgs:
    """In-memory shape of the YAML the engine hands to the runner."""

    # Data
    corpus_registry: Path | None
    latent_h5: Path | None
    fold: int
    input_latents: tuple[str, ...]
    target_latent: str
    max_patients_per_cohort: int | None

    # Architecture (paper-faithful — same MAISI 3-level config as T1C-RFlow).
    latent_channels: int
    cond_latents: int

    # PatchGAN discriminator.
    disc_ndf: int
    disc_num_layers: int

    # Pix2Pix-specific hyperparameters (Isola 2017 §3.2).
    lambda_l1: float
    lr_g: float
    lr_d: float
    beta1: float
    beta2: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    save_epoch_freq: int
    log_every: int
    num_workers: int
    use_amp: bool

    # Runtime
    gpu_id: int
    seed: int
    cohort_path_overrides: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model definitions — short enough to inline; mirror the vendored upstream's
# `GeneratorUNetWrapper` + `PatchDiscriminator3D` (see
# src/external/lpix2pix_3d/upstream/train_pix2pix_t1n_t2f.py:98-132).
# ---------------------------------------------------------------------------


class _GeneratorUNetWrapper(nn.Module):
    """Wrap :class:`DiffusionModelUNetMaisi` to ignore timesteps.

    Feeds ``t = zeros((B,), dtype=long)`` so the diffusion U-Net runs as a
    deterministic conditional generator. One forward pass maps the
    channel-concatenated conditioning latents to the predicted target
    latent.
    """

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        return self.unet(x, t)


class _PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator (~70³ receptive field for latents at 48³).

    Layer 0 has no normalisation (per Isola §6.1); subsequent layers stack
    ``Conv3d`` + ``InstanceNorm3d`` + ``LeakyReLU(0.2)`` with kernel=4,
    stride=2 (final non-output layer uses stride=1 to preserve patch
    resolution). The terminal head is a 1-channel ``Conv3d(k=4, s=1, p=1)``
    that produces patch-logits ``(B, 1, D', H', W')`` consumed by
    ``BCEWithLogitsLoss``.

    Parameters
    ----------
    in_channels:
        Discriminator input channels — for VENA's many-to-one Pix2Pix
        recipe ``cond_ch_total + target_ch = latent_ch*(1+cond_latents) =
        4*3 = 12``.
    ndf:
        Base number of filters (Isola 2017 §6.1 used 64).
    num_layers:
        Number of strided conv layers before the terminal head (default 4
        matches the vendored upstream).
    """

    def __init__(self, in_channels: int, ndf: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            stride = 2 if n < num_layers - 1 else 1
            layers += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm3d(ndf * nf_mult, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    """Seed Python, NumPy, and torch."""
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_generator(latent_channels: int, cond_latents: int) -> nn.Module:
    """Instantiate the generator at the **paper-faithful** MAISI U-Net config.

    Same architecture choice as the T1C-RFlow wrapper (3 levels,
    ``[128, 128, 256]`` channels, 2 residual blocks per level, no
    self-attention). The Pix2Pix recipe consumes the **conditioning latents
    only** at the input — there is no noisy target injection, no scheduler
    — so ``in_channels = latent_channels * cond_latents`` (NOT ``1 +
    cond_latents`` as in T1C-RFlow / DiT-3D).

    Returns the bare MONAI module wrapped in :class:`_GeneratorUNetWrapper`
    so callers receive a ``forward(x: Tensor) -> Tensor`` signature.
    """
    in_channels = latent_channels * cond_latents
    paper_kwargs: dict[str, Any] = {
        "spatial_dims": 3,
        "in_channels": in_channels,
        "out_channels": latent_channels,
        # Paper-faithful 3-level MAISI U-Net (mirrors T1C-RFlow runner).
        "num_channels": [128, 128, 256],
        "attention_levels": [False, False, False],
        # MAISI's SpatialAttention is always constructed; even when inactive
        # ``num_head_channels[i]`` must divide ``num_channels[i]``. 32 divides
        # both 128 and 256.
        "num_head_channels": [32, 32, 32],
        "num_res_blocks": 2,
        "use_flash_attention": False,
        "include_top_region_index_input": False,
        "include_bottom_region_index_input": False,
        "include_spacing_input": False,
        "num_class_embeds": None,
        "resblock_updown": True,
        "include_fc": True,
    }
    from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
        DiffusionModelUNetMaisi,
    )

    unet = DiffusionModelUNetMaisi(**paper_kwargs)
    n_unet = sum(p.numel() for p in unet.parameters())
    logger.info(
        "Built Pix2Pix generator (DiffusionModelUNetMaisi, paper-faithful): "
        "in_channels=%d out_channels=%d num_channels=%s num_res_blocks=%d "
        "params=%.2fM",
        in_channels,
        latent_channels,
        paper_kwargs["num_channels"],
        paper_kwargs["num_res_blocks"],
        n_unet / 1e6,
    )
    return _GeneratorUNetWrapper(unet)


def _build_discriminator(
    latent_channels: int, cond_latents: int, ndf: int, num_layers: int
) -> nn.Module:
    """Instantiate the 3D PatchGAN discriminator.

    Input channels = ``cond_ch_total + latent_channels`` because the
    discriminator sees ``cat([cond, target_or_fake], dim=1)`` (Isola §3.2,
    vendored ``train_pix2pix_t1n_t2f.py:191``).
    """
    in_channels = latent_channels * cond_latents + latent_channels
    disc = _PatchDiscriminator3D(in_channels=in_channels, ndf=ndf, num_layers=num_layers)
    n_disc = sum(p.numel() for p in disc.parameters())
    logger.info(
        "Built PatchDiscriminator3D: in_channels=%d ndf=%d num_layers=%d params=%.2fM",
        in_channels,
        ndf,
        num_layers,
        n_disc / 1e6,
    )
    return disc


def _build_dataset(cfg: _RunnerArgs, phase: str) -> Any:
    """Single- or multi-cohort dataset depending on which YAML field is set."""
    if cfg.corpus_registry is not None and cfg.latent_h5 is not None:
        raise Pix2PixRunnerError(
            "exactly one of {corpus_registry, latent_h5} must be set; got both"
        )
    if cfg.corpus_registry is not None:
        return MultiCohortPix2PixLatentDataset(
            corpus_registry=cfg.corpus_registry,
            fold=cfg.fold,
            phase=phase,
            input_latents=cfg.input_latents,
            target_latent=cfg.target_latent,
            max_patients_per_cohort=cfg.max_patients_per_cohort,
            path_overrides=cfg.cohort_path_overrides or None,
        )
    if cfg.latent_h5 is not None:
        return Pix2PixLatentDataset(
            latent_h5=cfg.latent_h5,
            fold=cfg.fold,
            phase=phase,
            input_latents=cfg.input_latents,
            target_latent=cfg.target_latent,
            max_patients=cfg.max_patients_per_cohort,
        )
    raise Pix2PixRunnerError("neither corpus_registry nor latent_h5 set — nothing to train on")


def _stack_conditioning(
    batch: dict[str, torch.Tensor],
    cond_keys: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Concatenate conditioning latents along the channel axis."""
    parts = [batch[f"z_{k}"].to(device, non_blocking=True) for k in cond_keys]
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train_lpix2pix_3d(cfg: SimpleNamespace, run_dir: Path) -> Path:
    """Train the 3D-Latent-Pix2Pix model and write checkpoints + CSV metrics.

    Parameters
    ----------
    cfg : SimpleNamespace
        Hyperparameters + data paths assembled by the routine engine. The
        fields are listed in :class:`_RunnerArgs`.
    run_dir : Path
        Output directory; will contain ``checkpoints/``, ``metrics/``,
        ``logs/``.

    Returns
    -------
    Path
        The ``run_dir`` (for chaining).
    """
    args = _RunnerArgs(
        corpus_registry=getattr(cfg, "corpus_registry", None),
        latent_h5=getattr(cfg, "latent_h5", None),
        fold=cfg.fold,
        input_latents=tuple(cfg.input_latents),
        target_latent=cfg.target_latent,
        max_patients_per_cohort=getattr(cfg, "max_patients_per_cohort", None),
        latent_channels=cfg.latent_channels,
        cond_latents=cfg.cond_latents,
        disc_ndf=cfg.disc_ndf,
        disc_num_layers=cfg.disc_num_layers,
        lambda_l1=cfg.lambda_l1,
        lr_g=cfg.lr_g,
        lr_d=cfg.lr_d,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        save_epoch_freq=cfg.save_epoch_freq,
        log_every=cfg.log_every,
        num_workers=cfg.num_workers,
        use_amp=cfg.use_amp,
        gpu_id=cfg.gpu_id,
        seed=cfg.seed,
        cohort_path_overrides={
            k: Path(v) for k, v in getattr(cfg, "cohort_path_overrides", {}).items()
        },
    )

    if args.latent_channels != 4:
        logger.warning(
            "latent_channels=%d (VENA MAISI-V2 = 4); ensure your VAE matches",
            args.latent_channels,
        )
    if args.cond_latents != len(args.input_latents):
        raise Pix2PixRunnerError(
            f"cond_latents={args.cond_latents} disagrees with "
            f"len(input_latents)={len(args.input_latents)}"
        )

    _seed_all(args.seed)
    device = (
        torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    )

    # -- Bookkeeping ----------------------------------------------------------
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "logs" / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(
        "3D-Latent-Pix2Pix trainer starting (Isola et al. 2017 + Eidex et al. "
        "2025 §4 baseline recipe) | device=%s seed=%d batch_size=%d "
        "max_epochs=%d patience=%d use_amp=%s lr_G=%.2e lr_D=%.2e "
        "betas=(%.2f,%.3f) wd=%.2e lambda_L1=%.1f",
        device,
        args.seed,
        args.batch_size,
        args.max_epochs,
        args.patience,
        args.use_amp,
        args.lr_g,
        args.lr_d,
        args.beta1,
        args.beta2,
        args.weight_decay,
        args.lambda_l1,
    )

    # -- Models + optimisers --------------------------------------------------
    netG = _build_generator(args.latent_channels, args.cond_latents).to(device)
    netD = _build_discriminator(
        args.latent_channels, args.cond_latents, args.disc_ndf, args.disc_num_layers
    ).to(device)

    optG = AdamW(
        netG.parameters(),
        lr=args.lr_g,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    optD = AdamW(
        netD.parameters(),
        lr=args.lr_d,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scalerG = GradScaler(enabled=args.use_amp and device.type == "cuda")
    scalerD = GradScaler(enabled=args.use_amp and device.type == "cuda")
    bce = nn.BCEWithLogitsLoss()

    # -- Data -----------------------------------------------------------------
    train_ds = _build_dataset(args, phase="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # -- CSV writers ----------------------------------------------------------
    step_csv = run_dir / "metrics" / "train_step.csv"
    epoch_csv = run_dir / "metrics" / "train_epoch.csv"
    step_fields = [
        "epoch",
        "global_step",
        "iter_in_epoch",
        "loss_g_total",
        "loss_g_adv",
        "loss_g_l1",
        "loss_d_real",
        "loss_d_fake",
        "lr_g",
        "lr_d",
        "step_seconds",
    ]
    epoch_fields = [
        "epoch",
        "loss_g_total_mean",
        "loss_g_adv_mean",
        "loss_g_l1_mean",
        "loss_d_real_mean",
        "loss_d_fake_mean",
        "wall_seconds",
    ]
    with step_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=step_fields).writeheader()
    with epoch_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=epoch_fields).writeheader()

    # -- Training loop --------------------------------------------------------
    # Model selection metric = epoch-mean ``loss_g_l1`` (the L1 reconstruction
    # component tracks output quality more directly than the BCE+L1 total,
    # whose D-driven adversarial term oscillates). Same selection metric
    # Isola et al. 2017 §6 reports.
    best_l1 = float("inf")
    best_epoch = -1
    no_improve = 0
    global_step = 0
    cond_keys = args.input_latents
    target_key = args.target_latent

    # Persist architecture metadata alongside every checkpoint so the
    # inference path can rebuild the model without consulting the YAML.
    arch_meta = {
        "latent_channels": args.latent_channels,
        "cond_latents": args.cond_latents,
        "disc_ndf": args.disc_ndf,
        "disc_num_layers": args.disc_num_layers,
    }

    for epoch in range(args.max_epochs):
        netG.train()
        netD.train()
        epoch_g_total: list[float] = []
        epoch_g_adv: list[float] = []
        epoch_g_l1: list[float] = []
        epoch_d_real: list[float] = []
        epoch_d_fake: list[float] = []
        epoch_t0 = time.perf_counter()

        for it, batch in enumerate(train_loader):
            step_t0 = time.perf_counter()
            tgt = batch[f"z_{target_key}"].to(device, non_blocking=True)
            cond = _stack_conditioning(batch, cond_keys, device)

            # -- update D --------------------------------------------------
            optD.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.use_amp and device.type == "cuda",
            ):
                fake = netG(cond)
                real_in = torch.cat([cond, tgt], dim=1)
                fake_in = torch.cat([cond, fake.detach()], dim=1)
                pred_real = netD(real_in)
                pred_fake = netD(fake_in)
                d_real = bce(pred_real, torch.ones_like(pred_real))
                d_fake = bce(pred_fake, torch.zeros_like(pred_fake))
                d_loss = 0.5 * (d_real + d_fake)

            if scalerD.is_enabled():
                scalerD.scale(d_loss).backward()
                scalerD.step(optD)
                scalerD.update()
            else:
                d_loss.backward()
                optD.step()

            # -- update G --------------------------------------------------
            optG.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.use_amp and device.type == "cuda",
            ):
                fake = netG(cond)
                fake_in = torch.cat([cond, fake], dim=1)
                pred_fake_g = netD(fake_in)
                g_adv = bce(pred_fake_g, torch.ones_like(pred_fake_g))
                g_l1 = F.l1_loss(fake, tgt)
                g_total = g_adv + args.lambda_l1 * g_l1

            if scalerG.is_enabled():
                scalerG.scale(g_total).backward()
                scalerG.step(optG)
                scalerG.update()
            else:
                g_total.backward()
                optG.step()

            g_total_val = float(g_total.item())
            g_adv_val = float(g_adv.item())
            g_l1_val = float(g_l1.item())
            d_real_val = float(d_real.item())
            d_fake_val = float(d_fake.item())

            epoch_g_total.append(g_total_val)
            epoch_g_adv.append(g_adv_val)
            epoch_g_l1.append(g_l1_val)
            epoch_d_real.append(d_real_val)
            epoch_d_fake.append(d_fake_val)
            global_step += 1
            step_dt = time.perf_counter() - step_t0

            if global_step % args.log_every == 0:
                logger.info(
                    "epoch=%d step=%d iter=%d g_total=%.4f g_adv=%.4f "
                    "g_l1=%.4f d_real=%.4f d_fake=%.4f dt=%.3fs",
                    epoch,
                    global_step,
                    it,
                    g_total_val,
                    g_adv_val,
                    g_l1_val,
                    d_real_val,
                    d_fake_val,
                    step_dt,
                )
            with step_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=step_fields).writerow(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "iter_in_epoch": it,
                        "loss_g_total": g_total_val,
                        "loss_g_adv": g_adv_val,
                        "loss_g_l1": g_l1_val,
                        "loss_d_real": d_real_val,
                        "loss_d_fake": d_fake_val,
                        "lr_g": args.lr_g,
                        "lr_d": args.lr_d,
                        "step_seconds": step_dt,
                    }
                )

        g_total_mean = float(np.mean(epoch_g_total)) if epoch_g_total else float("nan")
        g_adv_mean = float(np.mean(epoch_g_adv)) if epoch_g_adv else float("nan")
        g_l1_mean = float(np.mean(epoch_g_l1)) if epoch_g_l1 else float("nan")
        d_real_mean = float(np.mean(epoch_d_real)) if epoch_d_real else float("nan")
        d_fake_mean = float(np.mean(epoch_d_fake)) if epoch_d_fake else float("nan")
        wall = time.perf_counter() - epoch_t0

        with epoch_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=epoch_fields).writerow(
                {
                    "epoch": epoch,
                    "loss_g_total_mean": g_total_mean,
                    "loss_g_adv_mean": g_adv_mean,
                    "loss_g_l1_mean": g_l1_mean,
                    "loss_d_real_mean": d_real_mean,
                    "loss_d_fake_mean": d_fake_mean,
                    "wall_seconds": wall,
                }
            )
        logger.info(
            "epoch=%d g_total=%.4f g_adv=%.4f g_l1=%.4f d_real=%.4f d_fake=%.4f wall=%.1fs",
            epoch,
            g_total_mean,
            g_adv_mean,
            g_l1_mean,
            d_real_mean,
            d_fake_mean,
            wall,
        )

        # -- Checkpoints ------------------------------------------------------
        def _save(ckpt_path: Path) -> None:
            torch.save(
                {
                    "G_state_dict": netG.state_dict(),
                    "D_state_dict": netD.state_dict(),
                    "arch_meta": arch_meta,
                    "epoch": epoch,
                    "loss_g_total": g_total_mean,
                    "loss_g_l1": g_l1_mean,
                },
                ckpt_path,
            )

        _save(run_dir / "checkpoints" / "latest_net_pix2pix.pth")

        if g_l1_mean < best_l1:
            best_l1 = g_l1_mean
            best_epoch = epoch
            no_improve = 0
            _save(run_dir / "checkpoints" / "best_net_pix2pix.pth")
            logger.info("new best at epoch %d (g_l1=%.4f)", epoch, g_l1_mean)
        else:
            no_improve += 1

        if (epoch + 1) % args.save_epoch_freq == 0:
            _save(run_dir / "checkpoints" / f"epoch_{epoch}_net_pix2pix.pth")

        # -- Early stop -------------------------------------------------------
        if args.patience > 0 and no_improve >= args.patience:
            logger.info(
                "early stopping at epoch %d (no improvement for %d epochs; "
                "best_epoch=%d best_g_l1=%.4f)",
                epoch,
                no_improve,
                best_epoch,
                best_l1,
            )
            break

    # Sentinel for the watcher pattern (skill §6.1).
    logger.info("lpix2pix-3d-train completed")
    return run_dir

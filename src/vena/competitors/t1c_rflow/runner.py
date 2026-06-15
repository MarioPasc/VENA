"""T1C-RFlow training loop (Eidex *et al.* 2025, arXiv:2509.24194).

Mirrors ``src/external/t1c_rflow/upstream/train_rflow.py`` (SHA ``fc8314f6``)
verbatim on the load-bearing axes:

* U-Net is :class:`DiffusionModelUNetMaisi` from MONAI, instantiated from the
  vendored ``maisi/configs/config_maisi3d-rflow.json`` with
  ``diffusion_unet_def.in_channels`` overridden to ``latent_channels * (1 + len(cond)) = 12``
  (concat conditioning). Weights are randomly initialised — the paper trains
  from scratch.
* Scheduler is :class:`RFlowScheduler` directly (not VENA's :class:`RFlowEngine`,
  which does not expose ``use_timestep_transform`` / ``base_img_size_numel``).
  Kwargs match the paper:
  ``num_train_timesteps=1000``, ``use_discrete_timesteps=True``,
  ``sample_method="logit-normal"``, ``use_timestep_transform=True``,
  ``base_img_size_numel=64*64*48``, ``spatial_dim=3``.
* Loss is ``F.l1_loss(noise_pred, target - noise)`` — L1 on the velocity
  (paper Eq. 4, upstream ``train_rflow.py:207``). **L1, not L2.**
* Optimiser is ``AdamW(lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)``.
* Mixed precision is enabled by default (``torch.amp.autocast(dtype=fp16)``
  + ``GradScaler``) — the upstream script uses it; the paper does not state
  it. ``use_amp=False`` is a documented disable knob.
* **No EMA**, **no gradient clipping**, **no augmentation** — preserved from
  the paper and the upstream script.

What this runner adds on top of upstream
----------------------------------------
* Multi-cohort training over a VENA corpus registry (the upstream loader
  expects a single ``train/`` folder of ``.pt`` files).
* Per-step and per-epoch CSV logging (``metrics/train_step.csv``,
  ``metrics/train_epoch.csv``) matching pGAN-cGAN's schema.
* ``best`` / ``latest`` / ``epoch_<N>`` checkpoints with patience-based
  early stop on epoch-mean train loss (validation is asynchronous —
  exhaustive-val style — per the VENA paired-comparison axes).
* Self-contained log file (``logs/train.log``) attached as a ``FileHandler``
  at engine entry.
"""

from __future__ import annotations

import csv
import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import (
    MultiCohortT1CRFlowLatentDataset,
    T1CRFlowLatentDataset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class T1CRFlowRunnerError(Exception):
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

    # Hyperparams
    unet_arch_config: Path
    latent_channels: int
    cond_latents: int
    nfe_train_timesteps: int
    lr: float
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
# Vendored-config resolver (mirrors upstream/train_rflow.py:_resolve)
# ---------------------------------------------------------------------------

def _resolve(cfg: Any, root: dict[str, Any]) -> Any:
    """Recursively expand ``"@key"`` / ``"$@key"`` references in a MONAI mini-config.

    Equivalent to ``train_rflow.py``'s ``_resolve``: pulls primitive values
    out of the JSON's root namespace before passing kwargs to the class
    constructor.
    """
    if isinstance(cfg, dict):
        return {k: _resolve(v, root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_resolve(v, root) for v in cfg]
    if isinstance(cfg, str):
        if cfg.startswith("@"):
            return root[cfg[1:]]
        if cfg.startswith("$@"):
            return root[cfg[2:]]
    return cfg


def _instantiate(component_cfg: dict[str, Any], root_cfg: dict[str, Any]) -> Any:
    """Instantiate a MONAI-style ``{"_target_": "...", **kwargs}`` block.

    Mirrors ``train_rflow.py:instantiate`` so we follow the same loader the
    paper's authors used.
    """
    comp = dict(component_cfg)
    target = comp.pop("_target_")
    kwargs = _resolve(comp, root_cfg)
    module_path, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    """Seed Python, NumPy, and torch."""
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_unet(arch_config: Path, latent_channels: int, cond_latents: int) -> Any:
    """Instantiate the U-Net at the **paper-text** architecture.

    Per the VENA "paper-wins-over-code" policy (2026-06-15), the wrapper
    rebuilds the U-Net at the architecture Eidex *et al.* 2025 §3 explicitly
    describes::

        "We train a 3D diffusion UNet with a symmetrical encoder and
         decoder containing [128, 128, 256] channels at each layer and 2
         residual blocks per layer to approximate v_θ(z_t, t, c) with an
         L1 transport loss against the analytical target above, using
         timesteps drawn from the RFlow scheduler (1000 training
         timesteps) and conditioning vector c formed by channel-wise
         concatenation of the T1w and T2-FLAIR latents."

    Three levels, 2 res-blocks per level, **no self-attention** (not
    mentioned in the paper text), no class-conditioning, no spacing input.
    The vendored ``config_maisi3d-rflow.json`` (4-level + attention, 178.6
    M params) is **not** loaded here — it stays on disk for
    reproducibility of the *code* version, which is treated as a separate
    ablation row.

    The script-level override from upstream ``train_rflow.py:129``
    (``in_channels = latent_channels * 3``) is mirrored: for ``cond_latents=2``
    we get ``in_channels = 4 * (1 + 2) = 12``.
    """
    in_channels = latent_channels * (1 + cond_latents)
    paper_kwargs: dict[str, Any] = {
        "spatial_dims": 3,
        "in_channels": in_channels,
        "out_channels": latent_channels,
        # Paper text — Eidex et al. 2025 §3 verbatim.
        "num_channels": [128, 128, 256],
        "attention_levels": [False, False, False],
        # MAISI always constructs SpatialAttention per level; even when
        # ``attention_levels[i]=False`` the module needs ``num_head_channels[i]``
        # to be a positive divisor of ``num_channels[i]`` to avoid a divide-by-
        # zero at __init__. The blocks remain *inactive* (no attention is
        # computed). 32 divides 128 and 256 cleanly.
        "num_head_channels": [32, 32, 32],
        "num_res_blocks": 2,
        # Backbone-required toggles for the MAISI U-Net; the paper does not
        # use any of the class- / spacing- / region-conditioning paths.
        "use_flash_attention": False,
        "include_top_region_index_input": False,
        "include_bottom_region_index_input": False,
        "include_spacing_input": False,
        "num_class_embeds": None,
        "resblock_updown": True,
        "include_fc": True,
    }
    logger.info(
        "Building DiffusionModelUNetMaisi (PAPER-FAITHFUL): in_channels=%d, "
        "out_channels=%d, num_channels=%s, attention_levels=%s, "
        "num_res_blocks=%d. Vendored config at %s ignored on purpose "
        "(code = 4-level + attention; paper = 3-level conv-only).",
        in_channels,
        latent_channels,
        paper_kwargs["num_channels"],
        paper_kwargs["attention_levels"],
        paper_kwargs["num_res_blocks"],
        arch_config,
    )
    from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
        DiffusionModelUNetMaisi,
    )

    return DiffusionModelUNetMaisi(**paper_kwargs)


def _build_scheduler(num_train_timesteps: int) -> Any:
    """Instantiate the RFlow scheduler with the paper's exact kwargs.

    Hardcoded to match ``train_rflow.py:136-143``. ``base_img_size_numel`` is
    intentionally pinned to ``64*64*48`` even when the actual latents are
    ``60*60*40`` — this preserves the paper's resolution-aware time-warp
    distribution. Changing it would shift the timestep prior.
    """
    from monai.networks.schedulers.rectified_flow import RFlowScheduler

    return RFlowScheduler(
        num_train_timesteps=num_train_timesteps,
        use_discrete_timesteps=True,
        sample_method="logit-normal",
        use_timestep_transform=True,
        base_img_size_numel=64 * 64 * 48,
        spatial_dim=3,
    )


def _build_dataset(cfg: _RunnerArgs, phase: str) -> Any:
    """Single- or multi-cohort dataset depending on which YAML field is set."""
    if cfg.corpus_registry is not None and cfg.latent_h5 is not None:
        raise T1CRFlowRunnerError(
            "exactly one of {corpus_registry, latent_h5} must be set; got both"
        )
    if cfg.corpus_registry is not None:
        return MultiCohortT1CRFlowLatentDataset(
            corpus_registry=cfg.corpus_registry,
            fold=cfg.fold,
            phase=phase,
            input_latents=cfg.input_latents,
            target_latent=cfg.target_latent,
            max_patients_per_cohort=cfg.max_patients_per_cohort,
            path_overrides=cfg.cohort_path_overrides or None,
        )
    if cfg.latent_h5 is not None:
        return T1CRFlowLatentDataset(
            latent_h5=cfg.latent_h5,
            fold=cfg.fold,
            phase=phase,
            input_latents=cfg.input_latents,
            target_latent=cfg.target_latent,
            max_patients=cfg.max_patients_per_cohort,
        )
    raise T1CRFlowRunnerError(
        "neither corpus_registry nor latent_h5 set — nothing to train on"
    )


def _stack_conditioning(
    batch: dict[str, torch.Tensor], cond_keys: Sequence[str], device: torch.device
) -> torch.Tensor:
    """Concatenate conditioning latents along the channel axis."""
    parts = [batch[f"z_{k}"].to(device, non_blocking=True) for k in cond_keys]
    return torch.cat(parts, dim=1)


def train_t1c_rflow(cfg: SimpleNamespace, run_dir: Path) -> Path:
    """Train the T1C-RFlow U-Net and write checkpoints + CSV metrics.

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
        unet_arch_config=Path(cfg.unet_arch_config),
        latent_channels=cfg.latent_channels,
        cond_latents=cfg.cond_latents,
        nfe_train_timesteps=cfg.nfe_train_timesteps,
        lr=cfg.lr,
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

    if not args.unet_arch_config.is_file():
        raise T1CRFlowRunnerError(
            f"unet_arch_config not found at {args.unet_arch_config}"
        )
    if args.latent_channels != 4:
        logger.warning(
            "latent_channels=%d (paper uses 4); make sure your MAISI VAE matches",
            args.latent_channels,
        )
    if args.cond_latents != len(args.input_latents):
        raise T1CRFlowRunnerError(
            f"cond_latents={args.cond_latents} disagrees with "
            f"len(input_latents)={len(args.input_latents)}"
        )

    _seed_all(args.seed)
    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # -- Bookkeeping ----------------------------------------------------------
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Self-contained log file regardless of stdout redirection.
    fh = logging.FileHandler(run_dir / "logs" / "train.log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(fh)

    logger.info(
        "T1C-RFlow trainer starting (Eidex et al. 2025, arXiv:2509.24194) | "
        "device=%s seed=%d batch_size=%d max_epochs=%d patience=%d "
        "use_amp=%s lr=%.2e wd=%.2e",
        device, args.seed, args.batch_size, args.max_epochs, args.patience,
        args.use_amp, args.lr, args.weight_decay,
    )

    # -- Model + scheduler + optimiser ----------------------------------------
    unet = _build_unet(
        args.unet_arch_config, args.latent_channels, args.cond_latents
    ).to(device)
    scheduler = _build_scheduler(args.nfe_train_timesteps)
    optimiser = AdamW(
        unet.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")

    n_params = sum(p.numel() for p in unet.parameters())
    logger.info("U-Net parameter count: %.2fM", n_params / 1e6)

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
    step_fields = ["epoch", "global_step", "iter_in_epoch", "loss_rflow", "lr", "step_seconds"]
    epoch_fields = ["epoch", "loss_rflow_mean", "wall_seconds"]
    with step_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=step_fields).writeheader()
    with epoch_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=epoch_fields).writeheader()

    # -- Training loop --------------------------------------------------------
    best_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    global_step = 0
    cond_keys = args.input_latents
    target_key = args.target_latent

    for epoch in range(args.max_epochs):
        unet.train()
        epoch_losses: list[float] = []
        epoch_t0 = time.perf_counter()

        for it, batch in enumerate(train_loader):
            step_t0 = time.perf_counter()
            tgt = batch[f"z_{target_key}"].to(device, non_blocking=True)
            cond = _stack_conditioning(batch, cond_keys, device)
            noise = torch.randn_like(tgt)
            timesteps = scheduler.sample_timesteps(tgt)
            noisy = scheduler.add_noise(
                original_samples=tgt, noise=noise, timesteps=timesteps
            )
            # train_rflow.py:202 — concat = [noisy, cond...] along C axis.
            model_in = torch.cat([noisy, cond], dim=1)

            optimiser.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.use_amp and device.type == "cuda",
            ):
                pred = unet(model_in, timesteps)
                # train_rflow.py:207 — L1 on velocity target u_t = tgt - noise.
                loss = F.l1_loss(pred, tgt - noise)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()

            loss_val = float(loss.item())
            epoch_losses.append(loss_val)
            global_step += 1
            step_dt = time.perf_counter() - step_t0

            if global_step % args.log_every == 0:
                logger.info(
                    "epoch=%d step=%d iter=%d loss=%.4f dt=%.3fs",
                    epoch, global_step, it, loss_val, step_dt,
                )
            with step_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=step_fields).writerow(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "iter_in_epoch": it,
                        "loss_rflow": loss_val,
                        "lr": args.lr,
                        "step_seconds": step_dt,
                    }
                )

        epoch_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        wall = time.perf_counter() - epoch_t0
        with epoch_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=epoch_fields).writerow(
                {
                    "epoch": epoch,
                    "loss_rflow_mean": epoch_mean,
                    "wall_seconds": wall,
                }
            )
        logger.info(
            "epoch=%d mean_loss=%.4f wall=%.1fs",
            epoch, epoch_mean, wall,
        )

        # -- Checkpoints ------------------------------------------------------
        latest_ckpt = run_dir / "checkpoints" / "latest_net_unet.pth"
        torch.save(
            {
                "unet_state_dict": unet.state_dict(),
                "epoch": epoch,
                "train_loss": epoch_mean,
            },
            latest_ckpt,
        )

        if epoch_mean < best_loss:
            best_loss = epoch_mean
            best_epoch = epoch
            no_improve = 0
            best_ckpt = run_dir / "checkpoints" / "best_net_unet.pth"
            torch.save(
                {
                    "unet_state_dict": unet.state_dict(),
                    "epoch": epoch,
                    "train_loss": epoch_mean,
                },
                best_ckpt,
            )
            logger.info(
                "new best at epoch %d (train_loss=%.4f)", epoch, epoch_mean
            )
        else:
            no_improve += 1

        if (epoch + 1) % args.save_epoch_freq == 0:
            ep_ckpt = run_dir / "checkpoints" / f"epoch_{epoch}_net_unet.pth"
            torch.save(
                {
                    "unet_state_dict": unet.state_dict(),
                    "epoch": epoch,
                    "train_loss": epoch_mean,
                },
                ep_ckpt,
            )

        # -- Early stop -------------------------------------------------------
        if args.patience > 0 and no_improve >= args.patience:
            logger.info(
                "early stopping at epoch %d (no improvement for %d epochs; "
                "best_epoch=%d best_loss=%.4f)",
                epoch, no_improve, best_epoch, best_loss,
            )
            break

    # Sentinel for the watcher pattern (skill §6.1).
    logger.info("t1c-rflow-train completed")
    return run_dir

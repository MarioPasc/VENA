"""3D-DiT training loop (DiT backbone + rectified flow + L1 velocity loss).

The implementation mirrors ``vena.competitors.t1c_rflow.runner`` on every
non-backbone axis (scheduler, loss, optimiser, AMP, CSV-logging, checkpoint
cadence, patience-based early stop). The single axis that differs is the
backbone: we swap MONAI's ``DiffusionModelUNetMaisi`` for the vendored
``DiT3DWrapper`` (Peebles & Xie 2023 / Mo et al. 2023 architecture, 3D
adapted by the Eidex 2025 baseline at ``src/external/dit_3d/upstream/``).

Architecture choice (paper-faithful)
------------------------------------
DiT-B/4 in 3D:

- ``hidden_size = 768``
- ``depth = 12``
- ``num_heads = 12``
- ``patch_size = 4``
- ``input_size = (D, H, W)``  read from the latent shape at first batch
- ``in_channels = latent_channels × (1 + len(cond))`` for channel-concat
  conditioning (mirrors T1C-RFlow's ``train_rflow.py:129``).
- ``out_channels = latent_channels``
- ``num_classes = 1`` and ``class_dropout_prob = 0.0`` (no class label;
  conditioning is fully encoded in the channel concat).

This is Peebles & Xie 2023's standard "base" DiT size; patch_size=4
divides our multi-cohort latent grid ``(4, 48, 56, 48)`` cleanly along all
three axes.

Scheduler / loss / optimisation
-------------------------------
RFlow scheduler with the paper-pinned kwargs (mirrors T1C-RFlow runner
``_build_scheduler``); L1 velocity loss ``F.l1_loss(v_pred, z_T1c − z_noise)``;
``AdamW(lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)``; AMP enabled by
default (``torch.amp.GradScaler`` + ``autocast(fp16)``). No EMA, no
gradient clipping, no augmentation — same as T1C-RFlow.

Logging / checkpoints
---------------------
``best`` / ``latest`` / ``epoch_<N>`` checkpoints on epoch-mean train loss;
``metrics/train_step.csv`` + ``metrics/train_epoch.csv``; sentinel line
``"dit-3d-train completed"`` in the log so the watcher pattern works.

Citation: see ``vena/competitors/dit_3d/__init__.py``.
"""

from __future__ import annotations

import csv
import logging
import sys
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
    DiT3DLatentDataset,
    MultiCohortDiT3DLatentDataset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# Absolute path to the vendored DiT-3D upstream — the wrapper imports
# ``DiT3DWrapper`` from there at runtime.
_THIS = Path(__file__).resolve()
# src/vena/competitors/dit_3d/runner.py → repo_root = parents[4]
_REPO_ROOT = _THIS.parents[4]
_UPSTREAM_DIR = _REPO_ROOT / "src" / "external" / "dit_3d" / "upstream"


class DiT3DRunnerError(Exception):
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

    # Architecture (DiT-3D defaults match Peebles & Xie 2023 DiT-B/4).
    latent_channels: int
    cond_latents: int
    dit_hidden_size: int
    dit_depth: int
    dit_num_heads: int
    dit_patch_size: int
    dit_mlp_ratio: float

    # RFlow scheduler.
    nfe_train_timesteps: int

    # Optimisation.
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


def _import_dit_wrapper() -> Any:
    """Import ``DiT3DWrapper`` from the vendored upstream snapshot.

    We add the upstream directory to ``sys.path`` so the local
    ``dit3d_wrapper`` module finds ``dit3d`` (its sibling). The path is
    appended once and never popped — subsequent re-imports are no-ops.
    """
    if not _UPSTREAM_DIR.is_dir():
        raise DiT3DRunnerError(
            f"vendored upstream missing at {_UPSTREAM_DIR}; "
            "re-vendor per src/external/dit_3d/UPSTREAM.md"
        )
    path_str = str(_UPSTREAM_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    from dit3d_wrapper import DiT3DWrapper  # type: ignore[import-not-found]

    return DiT3DWrapper


def _build_dit3d(
    input_size: tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    hidden_size: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    mlp_ratio: float,
) -> Any:
    """Instantiate the DiT-3D model at the paper-faithful DiT-B/4 architecture.

    Validates the patch_size divides every spatial axis (DiT requires
    cleanly divisible patches; the positional embedding is a fixed-size
    buffer so failure here is a hard error, not a warning).
    """
    for axis_idx, axis_len in enumerate(input_size):
        if axis_len % patch_size != 0:
            raise DiT3DRunnerError(
                f"input_size[{axis_idx}]={axis_len} not divisible by "
                f"patch_size={patch_size}. The DiT-3D positional embedding "
                f"is fixed-size after init; resize the latent grid or "
                f"pick a compatible patch_size before training."
            )
    if hidden_size % 3 != 0:
        raise DiT3DRunnerError(
            f"DiT hidden_size={hidden_size} must be divisible by 3 (the 3D "
            f"sin-cos positional embedding splits the embedding dimension "
            f"into thirds — see dit3d.py:332)."
        )

    DiT3DWrapper = _import_dit_wrapper()
    model = DiT3DWrapper(
        in_channels=in_channels,
        out_channels=out_channels,
        input_size=input_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        class_dropout_prob=0.0,
        num_classes=1,
        learn_sigma=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built DiT-3D backbone: input_size=%s patch=%d hidden=%d depth=%d "
        "heads=%d in_channels=%d out_channels=%d params=%.2fM",
        input_size, patch_size, hidden_size, depth, num_heads,
        in_channels, out_channels, n_params / 1e6,
    )
    return model


def _build_scheduler(num_train_timesteps: int) -> Any:
    """RFlow scheduler with paper-pinned kwargs (mirrors T1C-RFlow runner).

    ``base_img_size_numel`` is pinned to ``64*64*48`` for parity with the
    Eidex 2025 implementation — the timestep prior depends on this and
    changing it would shift the noise schedule.
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
        raise DiT3DRunnerError(
            "exactly one of {corpus_registry, latent_h5} must be set; got both"
        )
    if cfg.corpus_registry is not None:
        return MultiCohortDiT3DLatentDataset(
            corpus_registry=cfg.corpus_registry,
            fold=cfg.fold,
            phase=phase,
            input_latents=cfg.input_latents,
            target_latent=cfg.target_latent,
            max_patients_per_cohort=cfg.max_patients_per_cohort,
            path_overrides=cfg.cohort_path_overrides or None,
        )
    if cfg.latent_h5 is not None:
        return DiT3DLatentDataset(
            latent_h5=cfg.latent_h5,
            fold=cfg.fold,
            phase=phase,
            input_latents=cfg.input_latents,
            target_latent=cfg.target_latent,
            max_patients=cfg.max_patients_per_cohort,
        )
    raise DiT3DRunnerError(
        "neither corpus_registry nor latent_h5 set — nothing to train on"
    )


def _stack_conditioning(
    batch: dict[str, torch.Tensor], cond_keys: Sequence[str], device: torch.device
) -> torch.Tensor:
    """Concatenate conditioning latents along the channel axis."""
    parts = [batch[f"z_{k}"].to(device, non_blocking=True) for k in cond_keys]
    return torch.cat(parts, dim=1)


def _peek_latent_shape(ds: Any, target_key: str) -> tuple[int, ...]:
    """Read one sample from the dataset to discover the latent grid shape.

    Returns the spatial shape ``(D, H, W)`` of the target latent. Raises
    ``DiT3DRunnerError`` if the dataset is empty.
    """
    if len(ds) == 0:
        raise DiT3DRunnerError("dataset is empty — cannot determine latent shape")
    sample = ds[0]
    z = sample[f"z_{target_key}"]
    if z.ndim != 4:
        raise DiT3DRunnerError(
            f"expected target latent rank 4 (C, D, H, W); got shape {tuple(z.shape)}"
        )
    return tuple(z.shape[1:])  # drop channel dim


def train_dit_3d(cfg: SimpleNamespace, run_dir: Path) -> Path:
    """Train the 3D-DiT and write checkpoints + CSV metrics.

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
        dit_hidden_size=cfg.dit_hidden_size,
        dit_depth=cfg.dit_depth,
        dit_num_heads=cfg.dit_num_heads,
        dit_patch_size=cfg.dit_patch_size,
        dit_mlp_ratio=cfg.dit_mlp_ratio,
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

    if args.latent_channels != 4:
        logger.warning(
            "latent_channels=%d (VENA MAISI-V2 = 4); ensure your VAE matches",
            args.latent_channels,
        )
    if args.cond_latents != len(args.input_latents):
        raise DiT3DRunnerError(
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

    fh = logging.FileHandler(run_dir / "logs" / "train.log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(fh)

    logger.info(
        "3D-DiT trainer starting (Peebles & Xie 2023 backbone, Eidex et al. "
        "2025 §4 baseline recipe) | device=%s seed=%d batch_size=%d "
        "max_epochs=%d patience=%d use_amp=%s lr=%.2e wd=%.2e",
        device, args.seed, args.batch_size, args.max_epochs, args.patience,
        args.use_amp, args.lr, args.weight_decay,
    )

    # -- Data first so we can read the latent shape ---------------------------
    train_ds = _build_dataset(args, phase="train")
    latent_grid = _peek_latent_shape(train_ds, args.target_latent)
    logger.info("Latent grid (D, H, W) = %s", latent_grid)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # -- Model + scheduler + optimiser ----------------------------------------
    in_channels = args.latent_channels * (1 + args.cond_latents)
    dit = _build_dit3d(
        input_size=latent_grid,
        in_channels=in_channels,
        out_channels=args.latent_channels,
        hidden_size=args.dit_hidden_size,
        depth=args.dit_depth,
        num_heads=args.dit_num_heads,
        patch_size=args.dit_patch_size,
        mlp_ratio=args.dit_mlp_ratio,
    ).to(device)
    scheduler = _build_scheduler(args.nfe_train_timesteps)
    optimiser = AdamW(
        dit.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")

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
        dit.train()
        epoch_losses: list[float] = []
        epoch_t0 = time.perf_counter()

        for it, batch in enumerate(train_loader):
            step_t0 = time.perf_counter()
            tgt = batch[f"z_{target_key}"].to(device, non_blocking=True)
            # Defensive shape check — the DiT positional embedding is fixed-size.
            if tuple(tgt.shape[2:]) != latent_grid:
                raise DiT3DRunnerError(
                    f"shape mismatch at step {global_step}: expected target "
                    f"spatial shape {latent_grid}, got {tuple(tgt.shape[2:])}. "
                    "Every cohort in the corpus must produce latents at the "
                    "same shape; bisect the corpus registry to find the "
                    "offending cohort."
                )
            cond = _stack_conditioning(batch, cond_keys, device)
            noise = torch.randn_like(tgt)
            timesteps = scheduler.sample_timesteps(tgt)
            noisy = scheduler.add_noise(
                original_samples=tgt, noise=noise, timesteps=timesteps
            )
            # Channel-concat conditioning: [noisy_target, cond_0, cond_1, ...].
            # Matches T1C-RFlow's train_rflow.py:202.
            model_in = torch.cat([noisy, cond], dim=1)

            optimiser.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.use_amp and device.type == "cuda",
            ):
                # DiT forward: (x, t, y=None). y=None → unconditional class
                # label; conditioning is fully in the channel concat.
                pred = dit(model_in, timesteps, y=None) \
                    if "y" in dit.forward.__code__.co_varnames \
                    else dit(model_in, timesteps)
                # L1 velocity loss — Eidex 2025 Eq. 4.
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
        # Persist the architecture kwargs alongside the state dict so the
        # inference path can rebuild the model without consulting the YAML.
        arch_meta = {
            "input_size": list(latent_grid),
            "in_channels": in_channels,
            "out_channels": args.latent_channels,
            "hidden_size": args.dit_hidden_size,
            "depth": args.dit_depth,
            "num_heads": args.dit_num_heads,
            "patch_size": args.dit_patch_size,
            "mlp_ratio": args.dit_mlp_ratio,
        }

        latest_ckpt = run_dir / "checkpoints" / "latest_net_dit.pth"
        torch.save(
            {
                "dit_state_dict": dit.state_dict(),
                "arch_meta": arch_meta,
                "epoch": epoch,
                "train_loss": epoch_mean,
            },
            latest_ckpt,
        )

        if epoch_mean < best_loss:
            best_loss = epoch_mean
            best_epoch = epoch
            no_improve = 0
            best_ckpt = run_dir / "checkpoints" / "best_net_dit.pth"
            torch.save(
                {
                    "dit_state_dict": dit.state_dict(),
                    "arch_meta": arch_meta,
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
            ep_ckpt = run_dir / "checkpoints" / f"epoch_{epoch}_net_dit.pth"
            torch.save(
                {
                    "dit_state_dict": dit.state_dict(),
                    "arch_meta": arch_meta,
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
    logger.info("dit-3d-train completed")
    return run_dir

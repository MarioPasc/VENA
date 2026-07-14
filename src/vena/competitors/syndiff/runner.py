"""Programmatic SynDiff training entrypoint.

Imports the patched vendored upstream (NCSNpp, ResNet generators, time-
conditional discriminators, EMA wrapper) via a sys.path shim, instantiates
the eight networks, and drives the training loop with VENA-style CSV
logging, per-epoch best/latest checkpointing, and a sentinel log line that
the smoke watcher matches.

Single-GPU only — distributed training and gradient broadcasting from
``train.py`` are intentionally stripped. ``--num_process_per_node 1`` was
the canonical README invocation; multi-GPU adds nothing for our budget and
risks DDP-init pain on heterogeneous platforms (server-3 / loginexa).
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SynDiffRunnerError(Exception):
    """Raised when the runner cannot proceed (missing upstream, bad cfg)."""


# ---------------------------------------------------------------------------
# Upstream import shim
# ---------------------------------------------------------------------------
_UPSTREAM_DIR = (Path(__file__).resolve().parent.parent.parent.parent
                 / "external" / "syndiff" / "upstream")


def _ensure_stylegan_ops() -> None:
    """Make ``utils.op`` importable, with or without a working CUDA toolchain.

    The vendored ``utils.op`` submodules call ``cpp_extension.load`` at *import*
    time, so importing them JIT-builds the StyleGAN2 fused kernels. Try that first
    — when it works we keep the fast fused path. When it does not (no ninja in the
    env, or an nvcc/gcc combination that will not compile the ``.cu`` sources, both
    of which we hit on Picasso), fall back to contingency C1 from
    ``src/external/syndiff/PATCHES.md``: the pure-PyTorch reference ops, which
    compute the same arithmetic and need no build.

    Must run before the vendored backbones are imported — they pull in ``utils.op``
    transitively.
    """
    try:
        import utils.op  # noqa: F401

        logger.debug("SynDiff fused CUDA ops built/loaded")
        return
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning(
            "SynDiff fused CUDA ops unavailable (%s: %s) — activating PATCHES.md "
            "contingency C1 (pure-PyTorch reference ops)",
            type(exc).__name__,
            str(exc).splitlines()[0] if str(exc) else "",
        )

    # A failed `load()` can leave a half-initialised entry behind; clear it so the
    # shim is what the backbones actually resolve.
    for name in [n for n in sys.modules if n == "utils.op" or n.startswith("utils.op.")]:
        del sys.modules[name]

    from .fused_ops_fallback import install as install_native_fused_ops

    install_native_fused_ops()


def _import_upstream():
    """Inject the vendored ``upstream/`` directory into ``sys.path`` and import.

    Returns a small ``SimpleNamespace`` with the eight model-factory handles
    plus the EMA wrapper, so the rest of the runner can pull from it without
    sprinkling ``sys.path`` mutations across functions.
    """
    if not _UPSTREAM_DIR.is_dir():
        raise SynDiffRunnerError(
            f"vendored SynDiff upstream not found at {_UPSTREAM_DIR}"
        )
    upstream = str(_UPSTREAM_DIR)
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    _ensure_stylegan_ops()
    try:
        from backbones.discriminator import Discriminator_large
        from backbones.generator_resnet import define_D, define_G
        from backbones.ncsnpp_generator_adagn import NCSNpp
        from utils.EMA import EMA
    except ImportError as exc:
        raise SynDiffRunnerError(
            f"failed to import vendored SynDiff modules from {_UPSTREAM_DIR}: {exc}"
        ) from exc
    return SimpleNamespace(
        NCSNpp=NCSNpp,
        define_G=define_G,
        define_D=define_D,
        Discriminator_large=Discriminator_large,
        EMA=EMA,
    )


# ---------------------------------------------------------------------------
# Diffusion schedule helpers (mirrored from train.py — kept module-local so
# our runner does not need to re-enter the upstream training entrypoint).
# ---------------------------------------------------------------------------
def _var_func_vp(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return 1.0 - torch.exp(2.0 * log_mean_coeff)


def _extract(input: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def _get_sigma_schedule(num_timesteps: int, beta_min: float, beta_max: float,
                        device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps_small = 1e-3
    t = np.arange(0, num_timesteps + 1, dtype=np.float64)
    t = t / num_timesteps
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small
    var = _var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device).to(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class _DiffusionCoefficients:
    def __init__(self, num_timesteps: int, beta_min: float, beta_max: float,
                 device: torch.device) -> None:
        self.sigmas, self.a_s, _ = _get_sigma_schedule(num_timesteps, beta_min, beta_max, device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def _q_sample_pairs(coeff: _DiffusionCoefficients, x_start: torch.Tensor,
                    t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x_start)
    x_t = _extract(coeff.a_s_cum, t, x_start.shape) * x_start \
        + _extract(coeff.sigmas_cum, t, x_start.shape) * noise
    noise_next = torch.randn_like(x_start)
    x_t_plus_one = _extract(coeff.a_s, t + 1, x_start.shape) * x_t \
        + _extract(coeff.sigmas, t + 1, x_start.shape) * noise_next
    return x_t, x_t_plus_one


class _PosteriorCoefficients:
    def __init__(self, num_timesteps: int, beta_min: float, beta_max: float,
                 device: torch.device) -> None:
        _, _, betas = _get_sigma_schedule(num_timesteps, beta_min, beta_max, device)
        betas = betas.type(torch.float32)[1:]
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.0], dtype=torch.float32, device=device),
             self.alphas_cumprod[:-1]), 0,
        )
        self.posterior_variance = betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def _sample_posterior(pos: _PosteriorCoefficients, x_0: torch.Tensor,
                      x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    mean = _extract(pos.posterior_mean_coef1, t, x_t.shape) * x_0 \
        + _extract(pos.posterior_mean_coef2, t, x_t.shape) * x_t
    log_var = _extract(pos.posterior_log_variance_clipped, t, x_t.shape)
    noise = torch.randn_like(x_t)
    nonzero_mask = (1 - (t == 0).type(torch.float32))
    return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise


# ---------------------------------------------------------------------------
# Argparse-style Namespace from VENA config
# ---------------------------------------------------------------------------
def _build_args(cfg, device: torch.device) -> SimpleNamespace:
    """Translate VENA SynDiff config → the Namespace NCSNpp / Discriminator expect."""
    return SimpleNamespace(
        # NCSNpp config (read by NCSNpp.__init__)
        not_use_tanh=False,
        centered=True,
        z_emb_dim=cfg.z_emb_dim,
        num_channels_dae=cfg.num_channels_dae,
        ch_mult=list(cfg.ch_mult),
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=tuple(cfg.attn_resolutions),
        dropout=cfg.dropout,
        resamp_with_conv=True,
        image_size=cfg.image_size,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="none",
        progressive_input="residual",
        progressive_combine="sum",
        embedding_type=cfg.embedding_type,
        fourier_scale=16.0,
        num_channels=2,                # NCSNpp input is cat(noisy, source) → 2-channel
        nz=cfg.nz,
        n_mlp=cfg.n_mlp,
        # Discriminator config
        t_emb_dim=cfg.t_emb_dim,
        ngf=cfg.ngf,
        # Diffusion schedule
        num_timesteps=cfg.num_timesteps,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
        # Training
        batch_size=cfg.batch_size,
        lr_g=cfg.lr_g,
        lr_d=cfg.lr_d,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        use_ema=cfg.use_ema,
        ema_decay=cfg.ema_decay,
        r1_gamma=cfg.r1_gamma,
        lazy_reg=cfg.lazy_reg,
        lambda_l1_loss=cfg.lambda_l1_loss,
        # Misc accessed by some upstream code paths
        seed=cfg.seed,
        device=str(device),
    )


# ---------------------------------------------------------------------------
# Network construction + best/latest checkpoint I/O
# ---------------------------------------------------------------------------
def _build_networks(upstream, args, device: torch.device, gpu_index: int):
    """Construct the 8 networks and move them to ``device``.

    Returns a SimpleNamespace with named handles. ``gen_diffusive_1`` is the
    target←source synthesiser — the model inference loads.
    """
    gen_diffusive_1 = upstream.NCSNpp(args).to(device)
    gen_diffusive_2 = upstream.NCSNpp(args).to(device)

    # ResNet generators are 1→1 channel (image-to-image), regardless of NCSNpp's
    # 2-channel input. We pass input_nc / output_nc explicitly so the upstream's
    # in-place `args.num_channels = 1` hack (train.py:245) is unnecessary.
    gen_non_diffusive_1to2 = upstream.define_G(
        input_nc=1, output_nc=1, ngf=64, netG="resnet_6blocks",
        norm="instance", use_dropout=False, init_type="normal",
        init_gain=0.02, gpu_ids=[gpu_index],
    )
    gen_non_diffusive_2to1 = upstream.define_G(
        input_nc=1, output_nc=1, ngf=64, netG="resnet_6blocks",
        norm="instance", use_dropout=False, init_type="normal",
        init_gain=0.02, gpu_ids=[gpu_index],
    )

    disc_diffusive_1 = upstream.Discriminator_large(
        nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim,
        act=nn.LeakyReLU(0.2),
    ).to(device)
    disc_diffusive_2 = upstream.Discriminator_large(
        nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim,
        act=nn.LeakyReLU(0.2),
    ).to(device)

    disc_non_diffusive_cycle1 = upstream.define_D(
        input_nc=1, ndf=64, which_model_netD="basic", n_layers_D=3,
        norm="instance", use_sigmoid=False, init_type="normal",
        init_gain=0.02, gpu_ids=[gpu_index],
    )
    disc_non_diffusive_cycle2 = upstream.define_D(
        input_nc=1, ndf=64, which_model_netD="basic", n_layers_D=3,
        norm="instance", use_sigmoid=False, init_type="normal",
        init_gain=0.02, gpu_ids=[gpu_index],
    )

    return SimpleNamespace(
        gen_diffusive_1=gen_diffusive_1,
        gen_diffusive_2=gen_diffusive_2,
        gen_non_diffusive_1to2=gen_non_diffusive_1to2,
        gen_non_diffusive_2to1=gen_non_diffusive_2to1,
        disc_diffusive_1=disc_diffusive_1,
        disc_diffusive_2=disc_diffusive_2,
        disc_non_diffusive_cycle1=disc_non_diffusive_cycle1,
        disc_non_diffusive_cycle2=disc_non_diffusive_cycle2,
    )


def _save_generator_set(nets, run_dir: Path, tag: str) -> None:
    """Save ``gen_diffusive_1`` + the three other generators under ``tag``.

    ``gen_diffusive_1`` is what inference loads. The other three are saved for
    completeness so a follow-on run can resume the full bidirectional cycle.
    """
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(nets.gen_diffusive_1.state_dict(), ckpt_dir / f"{tag}_gen_diffusive_1.pth")
    torch.save(nets.gen_diffusive_2.state_dict(), ckpt_dir / f"{tag}_gen_diffusive_2.pth")
    torch.save(nets.gen_non_diffusive_1to2.state_dict(),
               ckpt_dir / f"{tag}_gen_non_diffusive_1to2.pth")
    torch.save(nets.gen_non_diffusive_2to1.state_dict(),
               ckpt_dir / f"{tag}_gen_non_diffusive_2to1.pth")


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------
def train_syndiff(cfg, run_dir: Path) -> Path:
    """Run SynDiff training. Returns ``run_dir``.

    ``cfg`` is the flattened SynDiff training config (see engine.py); the runner
    does not need the Pydantic wrapper to keep the import surface small.
    """
    from .dataset import MultiCohortSynDiffSliceDataset, SynDiffSliceDataset

    run_dir = Path(run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "logs" / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("SynDiff runner — run_dir=%s", run_dir)

    # Determinism — seed both CPU and CUDA.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    gpu_index = cfg.gpu_ids[0] if cfg.gpu_ids else 0
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(gpu_index)
    logger.info("Using device %s", device)

    upstream = _import_upstream()
    args = _build_args(cfg, device)

    # ---- Build dataset(s) ----
    if getattr(cfg, "corpus_registry", None):
        overrides = {k: v for k, v in getattr(cfg, "cohort_path_overrides", {}).items()}
        train_ds = MultiCohortSynDiffSliceDataset(
            corpus_registry=cfg.corpus_registry,
            fold=cfg.fold,
            phase="train",
            target_modality=cfg.target_modality,
            source_modality=cfg.source_modality,
            image_size=cfg.image_size,
            min_brain_voxels=cfg.min_brain_voxels,
            max_patients_per_cohort=getattr(cfg, "max_patients_per_cohort", None),
            path_overrides=overrides,
        )
    else:
        train_ds = SynDiffSliceDataset(
            image_h5=cfg.image_h5,
            fold=cfg.fold,
            phase="train",
            target_modality=cfg.target_modality,
            source_modality=cfg.source_modality,
            image_size=cfg.image_size,
            min_brain_voxels=cfg.min_brain_voxels,
            max_patients=getattr(cfg, "max_train_patients", None),
        )

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=bool(cfg.gpu_ids),
        persistent_workers=cfg.num_workers > 0,
    )
    logger.info("Train loader: %d slices, batch=%d, workers=%d",
                len(train_ds), cfg.batch_size, cfg.num_workers)
    if len(train_ds) == 0:
        raise SynDiffRunnerError(
            "training dataset has 0 slices — check corpus_registry / fold / phase."
        )

    # ---- Build networks ----
    nets = _build_networks(upstream, args, device, gpu_index)
    logger.info("Networks instantiated (8 modules; gen_diffusive_1 is the target←source synthesiser)")

    # ---- Optimisers ----
    opt_gen_diff_1 = optim.Adam(nets.gen_diffusive_1.parameters(),
                                lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_gen_diff_2 = optim.Adam(nets.gen_diffusive_2.parameters(),
                                lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_gen_nd_12 = optim.Adam(nets.gen_non_diffusive_1to2.parameters(),
                               lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_gen_nd_21 = optim.Adam(nets.gen_non_diffusive_2to1.parameters(),
                               lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_disc_diff_1 = optim.Adam(nets.disc_diffusive_1.parameters(),
                                 lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_disc_diff_2 = optim.Adam(nets.disc_diffusive_2.parameters(),
                                 lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_disc_nd_c1 = optim.Adam(nets.disc_non_diffusive_cycle1.parameters(),
                                lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_disc_nd_c2 = optim.Adam(nets.disc_non_diffusive_cycle2.parameters(),
                                lr=args.lr_d, betas=(args.beta1, args.beta2))

    if args.use_ema:
        opt_gen_diff_1 = upstream.EMA(opt_gen_diff_1, ema_decay=args.ema_decay)
        opt_gen_diff_2 = upstream.EMA(opt_gen_diff_2, ema_decay=args.ema_decay)
        opt_gen_nd_12 = upstream.EMA(opt_gen_nd_12, ema_decay=args.ema_decay)
        opt_gen_nd_21 = upstream.EMA(opt_gen_nd_21, ema_decay=args.ema_decay)

    # ---- Schedulers ----
    sched_gen_diff_1 = optim.lr_scheduler.CosineAnnealingLR(opt_gen_diff_1, cfg.max_epochs, eta_min=1e-5)
    sched_gen_diff_2 = optim.lr_scheduler.CosineAnnealingLR(opt_gen_diff_2, cfg.max_epochs, eta_min=1e-5)
    sched_gen_nd_12 = optim.lr_scheduler.CosineAnnealingLR(opt_gen_nd_12, cfg.max_epochs, eta_min=1e-5)
    sched_gen_nd_21 = optim.lr_scheduler.CosineAnnealingLR(opt_gen_nd_21, cfg.max_epochs, eta_min=1e-5)
    sched_disc_diff_1 = optim.lr_scheduler.CosineAnnealingLR(opt_disc_diff_1, cfg.max_epochs, eta_min=1e-5)
    sched_disc_diff_2 = optim.lr_scheduler.CosineAnnealingLR(opt_disc_diff_2, cfg.max_epochs, eta_min=1e-5)
    sched_disc_nd_c1 = optim.lr_scheduler.CosineAnnealingLR(opt_disc_nd_c1, cfg.max_epochs, eta_min=1e-5)
    sched_disc_nd_c2 = optim.lr_scheduler.CosineAnnealingLR(opt_disc_nd_c2, cfg.max_epochs, eta_min=1e-5)

    # ---- Diffusion schedule ----
    coeff = _DiffusionCoefficients(cfg.num_timesteps, cfg.beta_min, cfg.beta_max, device)
    pos_coeff = _PosteriorCoefficients(cfg.num_timesteps, cfg.beta_min, cfg.beta_max, device)

    # ---- CSV writers ----
    step_csv = run_dir / "metrics" / "train_step.csv"
    epoch_csv = run_dir / "metrics" / "train_epoch.csv"
    step_fields = [
        "epoch", "global_step", "iter_in_epoch",
        "G_adv", "G_cycle_adv", "G_L1", "G_cycle",
        "D_diff", "D_cycle", "lr_g", "lr_d", "step_seconds",
    ]
    epoch_fields = [
        "epoch", "G_adv_mean", "G_cycle_adv_mean", "G_L1_mean", "G_cycle_mean",
        "D_diff_mean", "D_cycle_mean", "wall_seconds",
    ]
    with step_csv.open("w", newline="") as f_step, epoch_csv.open("w", newline="") as f_epoch:
        step_w = csv.DictWriter(f_step, fieldnames=step_fields)
        step_w.writeheader()
        epoch_w = csv.DictWriter(f_epoch, fieldnames=epoch_fields)
        epoch_w.writeheader()

        best_loss = float("inf")
        best_epoch = -1
        no_improve_epochs = 0
        stopped_early = False
        global_step = 0

        for epoch in range(1, cfg.max_epochs + 1):
            epoch_start = time.time()
            acc: dict[str, list[float]] = {
                "G_adv": [], "G_cycle_adv": [], "G_L1": [], "G_cycle": [],
                "D_diff": [], "D_cycle": [],
            }
            for it, (x1, x2) in enumerate(loader):
                t0 = time.time()
                # x1 = target (contrast1), x2 = source (contrast2)
                real_data1 = x1.to(device, non_blocking=True)
                real_data2 = x2.to(device, non_blocking=True)

                # ---- D step: diffusive discriminators (both directions) ----
                for p in nets.disc_diffusive_1.parameters(): p.requires_grad = True
                for p in nets.disc_diffusive_2.parameters(): p.requires_grad = True
                for p in nets.disc_non_diffusive_cycle1.parameters(): p.requires_grad = True
                for p in nets.disc_non_diffusive_cycle2.parameters(): p.requires_grad = True
                nets.disc_diffusive_1.zero_grad()
                nets.disc_diffusive_2.zero_grad()

                t1 = torch.randint(0, cfg.num_timesteps, (real_data1.size(0),), device=device)
                t2 = torch.randint(0, cfg.num_timesteps, (real_data2.size(0),), device=device)
                x1_t, x1_tp1 = _q_sample_pairs(coeff, real_data1, t1)
                x1_t.requires_grad = True
                x2_t, x2_tp1 = _q_sample_pairs(coeff, real_data2, t2)
                x2_t.requires_grad = True

                D1_real = nets.disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)
                D2_real = nets.disc_diffusive_2(x2_t, t2, x2_tp1.detach()).view(-1)
                errD1_real = F.softplus(-D1_real).mean()
                errD2_real = F.softplus(-D2_real).mean()
                errD_real = errD1_real + errD2_real
                errD_real.backward(retain_graph=True)

                # R1 gradient penalty — every step if lazy_reg is None/0, else every k.
                apply_r1 = (
                    cfg.lazy_reg is None or cfg.lazy_reg == 0
                    or (global_step % cfg.lazy_reg == 0)
                )
                if apply_r1:
                    grad1_real = torch.autograd.grad(
                        outputs=D1_real.sum(), inputs=x1_t, create_graph=True,
                    )[0]
                    grad1_pen = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad2_real = torch.autograd.grad(
                        outputs=D2_real.sum(), inputs=x2_t, create_graph=True,
                    )[0]
                    grad2_pen = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    (args.r1_gamma / 2 * grad1_pen + args.r1_gamma / 2 * grad2_pen).backward()

                latent_z1 = torch.randn(real_data1.size(0), args.nz, device=device)
                latent_z2 = torch.randn(real_data2.size(0), args.nz, device=device)
                x1_0_predict = nets.gen_non_diffusive_2to1(real_data2)
                x2_0_predict = nets.gen_non_diffusive_1to2(real_data1)
                x1_0_predict_diff = nets.gen_diffusive_1(
                    torch.cat((x1_tp1.detach(), x2_0_predict), dim=1), t1, latent_z1,
                )
                x2_0_predict_diff = nets.gen_diffusive_2(
                    torch.cat((x2_tp1.detach(), x1_0_predict), dim=1), t2, latent_z2,
                )
                x1_pos_sample = _sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
                x2_pos_sample = _sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)
                D1_fake = nets.disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
                D2_fake = nets.disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)
                errD1_fake = F.softplus(D1_fake).mean()
                errD2_fake = F.softplus(D2_fake).mean()
                (errD1_fake + errD2_fake).backward()
                opt_disc_diff_1.step()
                opt_disc_diff_2.step()
                errD_diff_total = (errD_real + errD1_fake + errD2_fake).item()

                # ---- D step: cycle discriminators ----
                nets.disc_non_diffusive_cycle1.zero_grad()
                nets.disc_non_diffusive_cycle2.zero_grad()
                D_c1_real = nets.disc_non_diffusive_cycle1(real_data1).view(-1)
                D_c2_real = nets.disc_non_diffusive_cycle2(real_data2).view(-1)
                errD_c_real = F.softplus(-D_c1_real).mean() + F.softplus(-D_c2_real).mean()
                errD_c_real.backward(retain_graph=True)
                x1_0_predict_d = nets.gen_non_diffusive_2to1(real_data2).detach()
                x2_0_predict_d = nets.gen_non_diffusive_1to2(real_data1).detach()
                D_c1_fake = nets.disc_non_diffusive_cycle1(x1_0_predict_d).view(-1)
                D_c2_fake = nets.disc_non_diffusive_cycle2(x2_0_predict_d).view(-1)
                errD_c_fake = F.softplus(D_c1_fake).mean() + F.softplus(D_c2_fake).mean()
                errD_c_fake.backward()
                opt_disc_nd_c1.step()
                opt_disc_nd_c2.step()
                errD_cycle_total = (errD_c_real + errD_c_fake).item()

                # ---- G step: all four generators ----
                for p in nets.disc_diffusive_1.parameters(): p.requires_grad = False
                for p in nets.disc_diffusive_2.parameters(): p.requires_grad = False
                for p in nets.disc_non_diffusive_cycle1.parameters(): p.requires_grad = False
                for p in nets.disc_non_diffusive_cycle2.parameters(): p.requires_grad = False
                nets.gen_diffusive_1.zero_grad()
                nets.gen_diffusive_2.zero_grad()
                nets.gen_non_diffusive_1to2.zero_grad()
                nets.gen_non_diffusive_2to1.zero_grad()

                t1 = torch.randint(0, cfg.num_timesteps, (real_data1.size(0),), device=device)
                t2 = torch.randint(0, cfg.num_timesteps, (real_data2.size(0),), device=device)
                x1_t, x1_tp1 = _q_sample_pairs(coeff, real_data1, t1)
                x2_t, x2_tp1 = _q_sample_pairs(coeff, real_data2, t2)
                latent_z1 = torch.randn(real_data1.size(0), args.nz, device=device)
                latent_z2 = torch.randn(real_data2.size(0), args.nz, device=device)
                x1_0_predict = nets.gen_non_diffusive_2to1(real_data2)
                x2_0_predict_cycle = nets.gen_non_diffusive_1to2(x1_0_predict)
                x2_0_predict = nets.gen_non_diffusive_1to2(real_data1)
                x1_0_predict_cycle = nets.gen_non_diffusive_2to1(x2_0_predict)

                x1_0_predict_diff = nets.gen_diffusive_1(
                    torch.cat((x1_tp1.detach(), x2_0_predict), dim=1), t1, latent_z1,
                )
                x2_0_predict_diff = nets.gen_diffusive_2(
                    torch.cat((x2_tp1.detach(), x1_0_predict), dim=1), t2, latent_z2,
                )
                x1_pos_sample = _sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
                x2_pos_sample = _sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)
                out1 = nets.disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
                out2 = nets.disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)
                errG_adv = F.softplus(-out1).mean() + F.softplus(-out2).mean()

                Dc1_fake = nets.disc_non_diffusive_cycle1(x1_0_predict).view(-1)
                Dc2_fake = nets.disc_non_diffusive_cycle2(x2_0_predict).view(-1)
                errG_cycle_adv = F.softplus(-Dc1_fake).mean() + F.softplus(-Dc2_fake).mean()

                errG_L1 = (F.l1_loss(x1_0_predict_diff[:, [0], :], real_data1)
                           + F.l1_loss(x2_0_predict_diff[:, [0], :], real_data2))
                errG_cycle = (F.l1_loss(x1_0_predict_cycle, real_data1)
                              + F.l1_loss(x2_0_predict_cycle, real_data2))
                errG = (args.lambda_l1_loss * errG_cycle
                        + errG_adv + errG_cycle_adv
                        + args.lambda_l1_loss * errG_L1)
                errG.backward()
                opt_gen_diff_1.step()
                opt_gen_diff_2.step()
                opt_gen_nd_12.step()
                opt_gen_nd_21.step()

                # ---- Per-step logging ----
                dt = time.time() - t0
                lr_g = opt_gen_diff_1.param_groups[0]["lr"]
                lr_d = opt_disc_diff_1.param_groups[0]["lr"]
                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "iter_in_epoch": it,
                    "G_adv": errG_adv.item(),
                    "G_cycle_adv": errG_cycle_adv.item(),
                    "G_L1": errG_L1.item(),
                    "G_cycle": errG_cycle.item(),
                    "D_diff": errD_diff_total,
                    "D_cycle": errD_cycle_total,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "step_seconds": dt,
                }
                step_w.writerow(row)
                f_step.flush()
                for k in acc:
                    acc[k].append(row[k])
                global_step += 1
                if global_step % cfg.log_every == 0:
                    logger.info(
                        "epoch=%d step=%d G_L1=%.4f G_cycle=%.4f G_adv=%.4f "
                        "D_diff=%.4f D_cycle=%.4f lr_g=%.2e dt=%.2fs",
                        epoch, global_step, row["G_L1"], row["G_cycle"], row["G_adv"],
                        row["D_diff"], row["D_cycle"], lr_g, dt,
                    )

            # ---- End-of-epoch ----
            wall = time.time() - epoch_start
            means = {f"{k}_mean": (sum(v) / len(v) if v else float("nan"))
                     for k, v in acc.items()}
            epoch_w.writerow({"epoch": epoch, "wall_seconds": wall, **means})
            f_epoch.flush()
            logger.info(
                "epoch %d done in %.1fs — G_L1=%.4f G_cycle=%.4f G_adv=%.4f "
                "D_diff=%.4f D_cycle=%.4f",
                epoch, wall, means["G_L1_mean"], means["G_cycle_mean"],
                means["G_adv_mean"], means["D_diff_mean"], means["D_cycle_mean"],
            )

            if not cfg.no_lr_decay:
                for s in (sched_gen_diff_1, sched_gen_diff_2, sched_gen_nd_12, sched_gen_nd_21,
                          sched_disc_diff_1, sched_disc_diff_2, sched_disc_nd_c1, sched_disc_nd_c2):
                    s.step()

            # ---- Checkpointing ----
            if epoch % cfg.save_epoch_freq == 0 or epoch == cfg.max_epochs:
                if args.use_ema:
                    for o in (opt_gen_diff_1, opt_gen_diff_2, opt_gen_nd_12, opt_gen_nd_21):
                        o.swap_parameters_with_ema(store_params_in_ema=True)
                _save_generator_set(nets, run_dir, tag=f"epoch_{epoch:04d}")
                if args.use_ema:
                    for o in (opt_gen_diff_1, opt_gen_diff_2, opt_gen_nd_12, opt_gen_nd_21):
                        o.swap_parameters_with_ema(store_params_in_ema=True)
                logger.info("saved periodic checkpoints at epoch %d", epoch)

            _save_generator_set(nets, run_dir, tag="latest")

            # Best selection — epoch G_L1 mean (forward L1 to GT on the target side).
            metric = means["G_L1_mean"]
            if metric < best_loss - 1e-6:
                best_loss = metric
                best_epoch = epoch
                no_improve_epochs = 0
                if args.use_ema:
                    for o in (opt_gen_diff_1, opt_gen_diff_2, opt_gen_nd_12, opt_gen_nd_21):
                        o.swap_parameters_with_ema(store_params_in_ema=True)
                _save_generator_set(nets, run_dir, tag="best")
                if args.use_ema:
                    for o in (opt_gen_diff_1, opt_gen_diff_2, opt_gen_nd_12, opt_gen_nd_21):
                        o.swap_parameters_with_ema(store_params_in_ema=True)
                logger.info("new best epoch %d (G_L1=%.4f) — saved best_gen_*.pth",
                            epoch, metric)
            else:
                no_improve_epochs += 1
                if cfg.patience > 0 and no_improve_epochs >= cfg.patience:
                    logger.info(
                        "early stopping at epoch %d (no improvement for %d epochs; "
                        "best was epoch %d at G_L1=%.4f)",
                        epoch, no_improve_epochs, best_epoch, best_loss,
                    )
                    stopped_early = True
                    break

    logger.info(
        "SynDiff training completed — %d epochs, %d steps "
        "(stopped_early=%s, best_epoch=%d, best_G_L1=%.4f)",
        epoch, global_step, stopped_early, best_epoch, best_loss,
    )
    # Sentinel — consumed by skill completion checks.
    logger.info("syndiff-train completed")
    return run_dir

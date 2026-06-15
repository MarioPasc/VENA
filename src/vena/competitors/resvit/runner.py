"""Programmatic training entrypoint for the vendored ResViT model.

Drives the two-stage curriculum from the paper (Dalmaz, Yurt, Çukur, IEEE TMI
2022, §III.B "Two-stage training procedure"):

1. **Stage 1 — CNN pretrain.** Train ``Res_CNN`` (residual-CNN encoder /
   decoder, no ART blocks, no transformer weights) on the cohort union for
   ``pretrain_niter + pretrain_niter_decay`` epochs. Saved as
   ``checkpoints/latest_pretrain_net_{G,D}.pth`` (separate filename so it
   survives stage 2's saves).
2. **Stage 2 — ART fine-tune.** Construct ``ResViT`` (same encoder / decoder
   with 9 ART blocks in the bottleneck). Warm-start the CNN weights from the
   stage-1 checkpoint (``pre_trained_resnet=1``) and load the ImageNet
   R50+ViT-B_16 transformer weights from the ``.npz`` cached under
   ``src/external/resvit/upstream/checkpoints/`` (``pre_trained_transformer=1``).
   Train for ``niter + niter_decay`` epochs. Best-by-epoch-mean-G_L1 and
   final-step weights saved as ``checkpoints/{best,latest}_net_{G,D}.pth``.

Both stages run in a single Python process; the trainer transitions between
them automatically. Stage 1 is unconditional (always runs at least one
epoch). Patience-based early stopping is applied to stage 2 only — the paper
treats stage 1 as a fixed-budget warm-up.

The vendored upstream ``util.visualizer.Visualizer`` is bypassed; losses and
LRs are written to ``metrics/{train_step,train_epoch}.csv`` directly. The
log handler writes ``logs/train.log`` for stdout-independent run capture.

Stage 1 / stage 2 metrics are written into the SAME CSV pair with a ``stage``
column distinguishing them, so the per-step and per-epoch loss trajectories
form one continuous record. Sentinel completion line ``"resvit-train
completed"`` is logged at the end of stage 2; the engine consumes it to flip
``decision.json.completed = true``.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class ResViTRunnerError(Exception):
    """Raised when the runner cannot proceed (missing ViT npz, bad cfg)."""


# ---------------------------------------------------------------------------
# Upstream import shim
# ---------------------------------------------------------------------------
_UPSTREAM_DIR = (Path(__file__).resolve().parent.parent.parent.parent
                 / "external" / "resvit" / "upstream")


def _push_upstream_path() -> list[str]:
    """Push the vendored upstream onto ``sys.path`` and return the prior state.

    The vendored ResViT uses relative imports (``from util import util``,
    ``from . import networks``). We push *temporarily* and restore via the
    returned snapshot in a ``finally`` block.
    """
    if not _UPSTREAM_DIR.is_dir():
        raise ResViTRunnerError(f"vendored ResViT upstream not found at {_UPSTREAM_DIR}")
    snapshot = list(sys.path)
    sys.path.insert(0, str(_UPSTREAM_DIR))
    return snapshot


def _set_vit_pretrained_path(vit_npz: Path) -> None:
    """Override the hardcoded ``Res-ViT-B_16.pretrained_path`` at runtime.

    Upstream's ``models/transformer_configs.py::get_resvit_b16_config`` hardcodes
    ``./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz`` (relative to the
    upstream repo's old directory layout). Patching the source file would
    bake a server-specific absolute path in; we instead mutate the
    ``ml_collections.ConfigDict`` field once at runtime. This is documented
    in ``src/external/resvit/PATCHES.md`` (§"Pre-trained ViT path override").
    """
    if not Path(vit_npz).is_file():
        raise ResViTRunnerError(
            f"ViT init checkpoint not found at {vit_npz}. Download with:\n"
            "  curl -sSL -o <path>/R50+ViT-B_16.npz "
            "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"
        )
    snapshot = _push_upstream_path()
    try:
        from models import residual_transformers  # type: ignore[import-not-found]
    finally:
        sys.path = snapshot
    residual_transformers.CONFIGS["Res-ViT-B_16"].pretrained_path = str(vit_npz)
    logger.info("Overrode CONFIGS['Res-ViT-B_16'].pretrained_path → %s", vit_npz)


def _import_resvit_model_factory():
    """Import the vendored ``models.create_model`` factory."""
    snapshot = _push_upstream_path()
    try:
        from models import create_model  # type: ignore[import-not-found]
    finally:
        sys.path = snapshot
    return create_model


# ---------------------------------------------------------------------------
# Option builder
# ---------------------------------------------------------------------------
def _build_opt(
    cfg,
    run_dir: Path,
    *,
    stage: str,
    which_model_netG: str,
    pre_trained_path: Path,
    pre_trained_resnet: int,
    pre_trained_transformer: int,
    niter: int,
    niter_decay: int,
    lr: float,
) -> SimpleNamespace:
    """Translate VENA config → the SimpleNamespace ResViT's BaseModel expects.

    Every flag the upstream ResViT model reads from its argparse must appear
    here. We do NOT call the upstream argparse parser — that would also
    create checkpoint dirs and an ``opt.txt`` we don't want.

    ``stage`` is a label (``"stage1_pretrain"`` / ``"stage2_finetune"``) carried
    along for log lines; it does not gate any upstream behaviour.
    """
    checkpoints_root = run_dir
    name = "checkpoints"
    (checkpoints_root / name).mkdir(parents=True, exist_ok=True)
    opt = SimpleNamespace(
        # Identification
        name=name,
        model=cfg.upstream_model,  # always "resvit_one" — see UPSTREAM.md table
        dataset_mode="aligned",
        which_direction="AtoB",
        # I/O channels
        input_nc=cfg.input_nc,
        output_nc=cfg.output_nc,
        # Architecture knobs
        ngf=cfg.ngf,
        ndf=cfg.ndf,
        n_layers_D=cfg.n_layers_D,
        norm=cfg.norm,
        init_type=cfg.init_type,
        no_dropout=cfg.no_dropout,
        which_model_netG=which_model_netG,
        which_model_netD="basic",
        # ViT / ART block config (consumed by define_G's 'res_cnn' & 'resvit' paths)
        vit_name=cfg.vit_name,
        fineSize=cfg.image_size,
        loadSize=cfg.image_size,
        pre_trained_path=str(pre_trained_path),
        pre_trained_transformer=pre_trained_transformer,
        pre_trained_resnet=pre_trained_resnet,
        # GAN
        no_lsgan=False,
        pool_size=cfg.pool_size,
        # Loss weights
        lambda_A=cfg.lambda_A,
        lambda_adv=cfg.lambda_adv,
        lambda_f=0.9,           # EMA momentum for D pool — paper default, unused for our purposes
        lambda_vgg=0.0,         # dead in upstream optimize_parameters; documented in UPSTREAM.md
        lambda_identity=0.0,
        # Optimizer
        lr=lr,
        beta1=cfg.beta1,
        trans_lr_coef=1.0,
        # LR schedule (linear decay over `niter_decay` after `niter`)
        lr_policy="lambda",
        lr_decay_iters=cfg.lr_decay_iters,
        epoch_count=1,
        niter=niter,
        niter_decay=niter_decay,
        # I/O paths (BaseModel uses checkpoints_dir + name to build save_dir)
        checkpoints_dir=str(checkpoints_root),
        # Continue/resume disabled — VENA always trains from scratch.
        continue_train=False,
        which_epoch="latest",
        isTrain=True,
        # GPU
        gpu_ids=cfg.gpu_ids,
        # Misc that ResViT reads but does not gate behaviour.
        batchSize=cfg.batchSize,
        serial_batches=False,
        # Sentinel (not consumed upstream)
        _vena_stage=stage,
    )
    return opt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _build_train_dataset(cfg) -> "Dataset":
    """Build either a single-cohort or multi-cohort training dataset.

    Mirrors ``vena.competitors.pgan_cgan.runner._build_train_dataset`` — same
    H5 schema (image-domain, multi-modality), same selector keys
    (``image_h5`` xor ``corpus_registry``).
    """
    from .dataset import MultiCohortImageSliceDataset, UCSFPDGMSliceDataset

    if getattr(cfg, "corpus_registry", None):
        overrides = {k: v for k, v in getattr(cfg, "cohort_path_overrides", {}).items()}
        return MultiCohortImageSliceDataset(
            corpus_registry=cfg.corpus_registry,
            fold=cfg.fold,
            phase="train",
            input_modalities=cfg.input_modalities,
            target_modality=cfg.target_modality,
            image_size=cfg.image_size,
            min_brain_voxels=cfg.min_brain_voxels,
            max_patients_per_cohort=getattr(cfg, "max_patients_per_cohort", None),
            path_overrides=overrides,
        )
    return UCSFPDGMSliceDataset(
        image_h5=cfg.image_h5,
        fold=cfg.fold,
        phase="train",
        input_modalities=cfg.input_modalities,
        target_modality=cfg.target_modality,
        image_size=cfg.image_size,
        min_brain_voxels=cfg.min_brain_voxels,
        max_patients=cfg.max_train_patients,
    )


# ---------------------------------------------------------------------------
# Single-stage training loop (called twice with different opts)
# ---------------------------------------------------------------------------
def _train_one_stage(
    *,
    cfg,
    opt: SimpleNamespace,
    run_dir: Path,
    stage: str,
    step_writer: csv.DictWriter,
    step_fh,
    epoch_writer: csv.DictWriter,
    epoch_fh,
    global_step_start: int,
    use_patience: bool,
    save_label_latest: str,
    save_label_best: str | None,
    max_slices: int | None = None,
) -> tuple[int, int, float]:
    """Run one stage of training.

    Returns ``(global_step_end, last_epoch, best_metric)``.

    Side effects:
    - Writes per-step rows to ``train_step.csv`` and per-epoch rows to
      ``train_epoch.csv`` (both already open by caller).
    - Saves checkpoints via the upstream ``model.save(label)`` method (which
      drops files into ``run_dir/checkpoints/<label>_net_{G,D}.pth``):
        * ``save_label_latest`` is written after every epoch.
        * ``save_label_best`` (if not None) is written on every G_L1 epoch
          mean improvement.
    """
    create_model = _import_resvit_model_factory()
    snapshot = _push_upstream_path()
    try:
        model = create_model(opt)
    finally:
        sys.path = snapshot

    train_ds = _build_train_dataset(cfg)
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
    logger.info("[%s] Train loader: %d slices, batch=%d, workers=%d",
                stage, len(train_ds), cfg.batchSize, cfg.num_workers)

    max_total_epochs = opt.niter + opt.niter_decay

    best_loss: float = float("inf")
    best_epoch: int = -1
    no_improve_epochs: int = 0
    global_step = global_step_start
    last_epoch = opt.epoch_count - 1
    # Paper-budget tracking — break out of both loops when this stage's
    # cumulative slice count crosses the cap.
    slices_seen_this_stage: int = 0
    slice_cap_tripped: bool = False
    if max_slices is not None:
        logger.info("[%s] paper-budget cap active: max_slices=%d", stage, max_slices)

    for epoch in range(opt.epoch_count, max_total_epochs + 1):
        epoch_start = time.time()
        acc: dict[str, list[float]] = {"G_GAN": [], "G_L1": [], "D_real": [], "D_fake": []}

        for it, batch in enumerate(loader):
            t0 = time.time()
            model.set_input(batch)
            model.optimize_parameters()
            errors = model.get_current_errors()
            lr = model.optimizers[0].param_groups[0]["lr"]
            dt = time.time() - t0

            step_writer.writerow({
                "stage": stage,
                "epoch": epoch,
                "global_step": global_step,
                "iter_in_epoch": it,
                "G_GAN": float(errors["G_GAN"]),
                "G_L1": float(errors["G_L1"]),
                "D_real": float(errors["D_real"]),
                "D_fake": float(errors["D_fake"]),
                "lr": float(lr),
                "step_seconds": dt,
            })
            step_fh.flush()
            for k in acc:
                acc[k].append(float(errors[k]))
            global_step += 1
            # Slice budget = batch_size per step (drop_last=True; B is constant).
            slices_seen_this_stage += int(batch["A"].shape[0])
            if global_step % cfg.log_every == 0:
                logger.info(
                    "[%s] epoch=%d step=%d slices=%d G_L1=%.4f G_GAN=%.4f D_real=%.4f "
                    "D_fake=%.4f lr=%.2e dt=%.2fs",
                    stage, epoch, global_step, slices_seen_this_stage,
                    errors["G_L1"], errors["G_GAN"], errors["D_real"], errors["D_fake"], lr, dt,
                )
            if max_slices is not None and slices_seen_this_stage >= max_slices:
                logger.info(
                    "[%s] paper-budget cap reached at step=%d (slices_seen=%d ≥ %d) "
                    "— breaking out of stage at epoch=%d, iter=%d",
                    stage, global_step, slices_seen_this_stage, max_slices, epoch, it,
                )
                slice_cap_tripped = True
                break

        wall = time.time() - epoch_start
        means = {f"{k}_mean": (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
        epoch_writer.writerow({
            "stage": stage,
            "epoch": epoch,
            "wall_seconds": wall,
            **means,
        })
        epoch_fh.flush()
        logger.info(
            "[%s] epoch %d done in %.1fs — G_L1=%.4f G_GAN=%.4f slices_total=%d",
            stage, epoch, wall, means["G_L1_mean"], means["G_GAN_mean"],
            slices_seen_this_stage,
        )

        # Always overwrite the stage's "latest" checkpoint.
        model.save(save_label_latest)
        model.update_learning_rate()
        last_epoch = epoch

        # Best-G tracking + patience-based early stopping (stage 2 only).
        metric = means["G_L1_mean"]
        if metric < best_loss - 1e-6:
            best_loss = metric
            best_epoch = epoch
            no_improve_epochs = 0
            if save_label_best is not None:
                model.save(save_label_best)
                logger.info("[%s] new best epoch %d (G_L1=%.4f) — saved %s_net_{G,D}.pth",
                            stage, epoch, metric, save_label_best)
        else:
            no_improve_epochs += 1
            if use_patience and cfg.patience > 0 and no_improve_epochs >= cfg.patience:
                logger.info(
                    "[%s] early stopping at epoch %d (no improvement for %d epochs; "
                    "best was epoch %d at G_L1=%.4f)",
                    stage, epoch, no_improve_epochs, best_epoch, best_loss,
                )
                break

        if slice_cap_tripped:
            # We already saved this partial epoch's `latest_` checkpoint above —
            # so the run still has a deliverable. Exit the epoch loop.
            break

    # Free GPU before the next stage builds new networks.
    del model
    del loader
    del train_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_step, last_epoch, best_loss


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def train_resvit(cfg, run_dir: Path) -> Path:
    """Run ResViT two-stage training. Returns ``run_dir``."""
    run_dir = Path(run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "logs" / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("ResViT runner — run_dir=%s", run_dir)
    logger.info("ResViT runner — stages: pretrain (%d+%d ep) → finetune (%d+%d ep)",
                cfg.pretrain_niter, cfg.pretrain_niter_decay, cfg.niter, cfg.niter_decay)

    # GPU placement — ResViT's BaseModel uses torch.cuda.set_device on gpu_ids[0].
    if cfg.gpu_ids:
        torch.cuda.set_device(cfg.gpu_ids[0])

    # Resolve ViT init checkpoint absolute path (default: vendored copy).
    vit_npz = Path(cfg.vit_init_npz).resolve()
    _set_vit_pretrained_path(vit_npz)

    step_csv = run_dir / "metrics" / "train_step.csv"
    epoch_csv = run_dir / "metrics" / "train_epoch.csv"
    step_fields = ["stage", "epoch", "global_step", "iter_in_epoch",
                   "G_GAN", "G_L1", "D_real", "D_fake", "lr", "step_seconds"]
    epoch_fields = ["stage", "epoch",
                    "G_GAN_mean", "G_L1_mean", "D_real_mean", "D_fake_mean", "wall_seconds"]

    with step_csv.open("w", newline="") as step_fh, epoch_csv.open("w", newline="") as epoch_fh:
        step_writer = csv.DictWriter(step_fh, fieldnames=step_fields)
        step_writer.writeheader()
        epoch_writer = csv.DictWriter(epoch_fh, fieldnames=epoch_fields)
        epoch_writer.writeheader()

        # -------------------- Stage 1: CNN pretrain --------------------
        opt_s1 = _build_opt(
            cfg, run_dir,
            stage="stage1_pretrain",
            which_model_netG="res_cnn",
            pre_trained_path=Path("/dev/null"),  # not consumed by res_cnn branch
            pre_trained_resnet=0,
            pre_trained_transformer=0,
            niter=cfg.pretrain_niter,
            niter_decay=cfg.pretrain_niter_decay,
            lr=cfg.pretrain_lr,
        )
        logger.info("===== Stage 1: CNN pretrain (Res_CNN, lr=%.2e) =====", cfg.pretrain_lr)
        global_step, last_epoch_s1, best_loss_s1 = _train_one_stage(
            cfg=cfg, opt=opt_s1, run_dir=run_dir,
            stage="stage1_pretrain",
            step_writer=step_writer, step_fh=step_fh,
            epoch_writer=epoch_writer, epoch_fh=epoch_fh,
            global_step_start=0,
            use_patience=False,
            save_label_latest="latest_pretrain",
            save_label_best=None,   # stage 1 has no "best" — only the final hand-off.
            max_slices=cfg.pretrain_max_slices,
        )
        stage1_ckpt = run_dir / "checkpoints" / "latest_pretrain_net_G.pth"
        if not stage1_ckpt.is_file():
            raise ResViTRunnerError(
                f"stage 1 finished but {stage1_ckpt} was not written"
            )
        logger.info("Stage 1 done — last_epoch=%d, ckpt=%s", last_epoch_s1, stage1_ckpt)

        # -------------------- Stage 2: ART fine-tune --------------------
        opt_s2 = _build_opt(
            cfg, run_dir,
            stage="stage2_finetune",
            which_model_netG="resvit",
            pre_trained_path=stage1_ckpt,
            pre_trained_resnet=1,
            pre_trained_transformer=1,
            niter=cfg.niter,
            niter_decay=cfg.niter_decay,
            lr=cfg.lr,
        )
        logger.info("===== Stage 2: ART fine-tune (ResViT, lr=%.2e, warm-start=%s) =====",
                    cfg.lr, stage1_ckpt.name)
        global_step, last_epoch_s2, best_loss_s2 = _train_one_stage(
            cfg=cfg, opt=opt_s2, run_dir=run_dir,
            stage="stage2_finetune",
            step_writer=step_writer, step_fh=step_fh,
            epoch_writer=epoch_writer, epoch_fh=epoch_fh,
            global_step_start=global_step,
            use_patience=True,
            save_label_latest="latest",
            save_label_best="best",
            max_slices=cfg.max_slices,
        )

    logger.info("ResViT training completed — stage 1: %d epochs (last G_L1=%.4f), "
                "stage 2: %d epochs (best G_L1=%.4f)",
                last_epoch_s1, best_loss_s1, last_epoch_s2, best_loss_s2)
    # Sentinel string consumed by skill completion checks.
    logger.info("resvit-train completed")
    return run_dir

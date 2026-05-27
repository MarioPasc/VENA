"""LightningModule wrapping trunk + ControlNet + RFlow + composite loss.

Trunk and (during validation) the VAE are frozen; only ControlNet parameters
are trained. The optimiser is constructed over ``self.controlnet.parameters()``
only.

The training step follows MAISI-v2's ControlNet recipe:

    down_residuals, mid_residual = controlnet(x_t, t, c_orig, class_labels)
    v = trunk(
        x_t, t,
        class_labels=class_labels,
        spacing_tensor=spacing,
        down_block_additional_residuals=down_residuals,
        mid_block_additional_residual=mid_residual,
    )
    loss = composite(LossInputs(..., v_orig=v, v_perturb=optional))

The validation step:

* runs the EMA shadow model on a sampler (default Euler, NFE in {per-epoch,
  sweep_nfes}),
* computes region-masked latent metrics (always) and image metrics (decoded
  through the frozen VAE on the per-epoch NFE and on every sweep NFE),
* populates buffers consumed by callbacks (`val_csv`, `qualitative`,
  `nfe_timing`).
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# PyTorch 2.6+ enforces ``weights_only=True`` on ``torch.load`` by default,
# which rejects unknown picklable globals. Lightning's checkpoint round-trip
# stores our :meth:`on_save_checkpoint` payload including the NumPy RNG state
# tuple (``np.random.get_state()``), whose internals reference
# ``numpy._core.multiarray._reconstruct`` and an ``ndarray`` dtype. We
# allowlist the safe globals once at module import so resume just works.
def _register_safe_globals() -> None:
    try:
        from numpy._core.multiarray import _reconstruct  # noqa: PLC0415
        torch.serialization.add_safe_globals(
            [_reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
        )
    except Exception:  # noqa: BLE001 — best-effort; older numpy / torch is fine
        pass


_register_safe_globals()

from ..controlnet.base import AbstractControlNet
from ..controlnet.conditioning import ConditioningAssembler, ConditioningSpec
from ..controlnet.losses import CompositeLoss, LossInputs, build_loss
from ..controlnet.maisi_controlnet import MaisiControlNet
from ..ema import WarmupEMA
from ..inference import EulerSampler
from ..maisi.config import TrunkConfig
from ..maisi.trunk import TrunkHandle, load_trunk
from ..metrics import ImageMetrics, LatentMetrics, RegionMasks, RegionResolver
from ..sampler.rflow import RFlowEngine

logger = logging.getLogger(__name__)


REGION_NAMES: tuple[str, ...] = ("full", "wt", "bg", "vessel")
REGION_TO_RESOLVER_KEY: dict[str, str] = {
    "full": "brain",   # full-brain mask
    "wt": "wt",
    "bg": "bg",
    "vessel": "vessel",
}


class FMLightningModule(pl.LightningModule):
    """End-to-end FM training step (ControlNet only)."""

    def __init__(
        self,
        trunk_config: TrunkConfig,
        conditioning_specs: list[str | ConditioningSpec],
        stage: str = "S1",
        loss_cfg: dict[str, Any] | None = None,
        perturb_keys: Iterable[str] | None = None,
        controlnet_arch_overrides: dict[str, Any] | None = None,
        optim_cfg: dict[str, Any] | None = None,
        rflow_cfg: dict[str, Any] | None = None,
        ema_cfg: dict[str, Any] | None = None,
        region_resolver: RegionResolver | None = None,
        validation_cfg: dict[str, Any] | None = None,
        vae_decoder: Any | None = None,
        nan_tolerance: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        # Lightning saves these into checkpoint hparams. We exclude unpicklables.
        self.save_hyperparameters(
            ignore=["trunk_config", "region_resolver", "vae_decoder"]
        )

        self.trunk_config = trunk_config
        self.stage = stage
        self.perturb_keys: set[str] = set(perturb_keys or ()) if perturb_keys else {"wt"}

        self._trunk_handle: TrunkHandle | None = None
        self.conditioning = ConditioningAssembler(conditioning_specs)
        cond_in = self.conditioning.total_channels
        logger.info("FMLightningModule: conditioning_total_channels=%d", cond_in)

        self.controlnet: AbstractControlNet = MaisiControlNet(
            conditioning_in_channels=cond_in,
            arch_overrides=controlnet_arch_overrides or {},
        )

        self.composite: CompositeLoss = build_loss(stage, loss_cfg or {})
        self.rflow = RFlowEngine(**(rflow_cfg or {}))
        self.optim_cfg: dict[str, Any] = optim_cfg or {}
        self.ema_cfg: dict[str, Any] = ema_cfg or {}

        # EMA must be built in __init__ so its parameters exist by the time
        # Lightning's checkpoint load_state_dict runs (it loads *before*
        # setup()).
        self.ema: WarmupEMA = WarmupEMA(self.controlnet, **self.ema_cfg)

        # Validation/region wiring.
        self.region_resolver = region_resolver
        self.validation_cfg: dict[str, Any] = validation_cfg or {}
        self.vae_decoder = vae_decoder
        self.latent_metrics = LatentMetrics()
        self.image_metrics: ImageMetrics | None = None  # built lazily if vae_decoder set

        # Buffers consumed by callbacks.
        self._val_accumulator: dict[tuple[int, str], dict[str, Any]] = {}
        self._qualitative_buffer: dict[tuple[str, int], torch.Tensor] = {}
        self._nfe_timing_buffer: list[dict[str, Any]] = []

        # NaN guard counters.
        nt = nan_tolerance or {}
        self._nan_max_in_window = int(nt.get("max_in_window", 10))
        self._nan_window = int(nt.get("window_steps", 1000))
        self._nan_history: list[int] = []  # step indices where NaN occurred

        # Step-time tracking (logged per training step).
        self._step_t0: float | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if self._trunk_handle is None:
            self._setup_trunk_and_controlnet()
        # Move EMA shadow to the same device as the live model.
        self.ema = self.ema.to(self.device)
        if self.image_metrics is None and self.vae_decoder is not None:
            self.image_metrics = ImageMetrics()

    def _setup_trunk_and_controlnet(self) -> None:
        ckpt = Path(self.trunk_config.checkpoint)
        arch_json = (
            Path(self.trunk_config.arch_json) if self.trunk_config.arch_json else None
        )
        self._trunk_handle = load_trunk(
            checkpoint_path=ckpt,
            device=self.device,
            arch_config=arch_json,
            arch_overrides=self.trunk_config.arch_overrides or None,
        )
        trunk_sd = self._trunk_handle.model.state_dict()
        self.controlnet.init_from_trunk(trunk_sd)
        self.controlnet.zero_init_output_projections()
        self.controlnet = self.controlnet.to(self.device)
        logger.info(
            "FMLightningModule.setup: trunk on %s (sha=%s) controlnet on %s",
            self._trunk_handle.device,
            self._trunk_handle.checkpoint_sha256[:12],
            self.device,
        )

    @property
    def trunk(self) -> torch.nn.Module:
        if self._trunk_handle is None:
            raise RuntimeError("trunk not loaded — call setup() first")
        return self._trunk_handle.model

    # ------------------------------------------------------------------
    # Padding helpers (trunk requires dims divisible by 8).
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor, multiple: int = 8
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        sizes = x.shape[-3:]
        pad_h = (multiple - sizes[0] % multiple) % multiple
        pad_w = (multiple - sizes[1] % multiple) % multiple
        pad_d = (multiple - sizes[2] % multiple) % multiple
        if pad_h == 0 and pad_w == 0 and pad_d == 0:
            return x, (0, 0, 0)
        padded = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        return padded, (pad_h, pad_w, pad_d)

    @staticmethod
    def _unpad(x: torch.Tensor, pad: tuple[int, int, int]) -> torch.Tensor:
        pad_h, pad_w, pad_d = pad
        if pad_h == 0 and pad_w == 0 and pad_d == 0:
            return x
        return x[..., : x.shape[-3] - pad_h, : x.shape[-2] - pad_w, : x.shape[-1] - pad_d]

    def _trunk_forward(
        self,
        controlnet: AbstractControlNet,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        class_labels: torch.Tensor,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        x_t_p, pad = self._pad_to_multiple(x_t, multiple=8)
        cond_p, _ = self._pad_to_multiple(cond, multiple=8)
        down_res, mid_res = controlnet(
            x=x_t_p,
            timesteps=timesteps,
            controlnet_cond=cond_p,
            class_labels=class_labels,
        )
        v_p = self.trunk(
            x=x_t_p,
            timesteps=timesteps,
            class_labels=class_labels,
            spacing_tensor=spacing,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )
        return self._unpad(v_p, pad)

    # ------------------------------------------------------------------
    # Training step.
    # ------------------------------------------------------------------

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self._step_t0 = time.perf_counter()

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor | None:
        x1 = batch["z_t1c"]
        B = x1.shape[0]
        device = x1.device

        x0 = torch.randn_like(x1)
        timesteps = self.rflow.sample_timesteps(x1).to(device)
        x_t = self.rflow.add_noise(x1, x0, timesteps)
        u_target = self.rflow.target_velocity(x1, x0)

        class_labels = self.trunk_config.make_class_labels(B, device)
        spacing = self.trunk_config.make_spacing_tensor(B, device)

        cond_orig = self.conditioning(batch)
        v_orig = self._trunk_forward(
            self.controlnet, x_t, timesteps, cond_orig, class_labels, spacing
        )

        v_perturb: torch.Tensor | None = None
        if self.composite.requires_perturbed_pass:
            cond_perturb = self.conditioning(batch, perturb_keys=self.perturb_keys)
            v_perturb = self._trunk_forward(
                self.controlnet,
                x_t, timesteps, cond_perturb, class_labels, spacing,
            )

        inputs = LossInputs(
            x_clean=x1,
            noise=x0,
            x_t=x_t,
            timesteps=timesteps,
            u_target=u_target,
            v_orig=v_orig,
            v_perturb=v_perturb,
            m_wt=batch.get("m_wt"),
        )
        total, per_term = self.composite(inputs)

        # NaN guard.
        if not torch.isfinite(total):
            self._nan_history.append(int(self.global_step))
            self._nan_history = [s for s in self._nan_history if self.global_step - s <= self._nan_window]
            logger.error(
                "NaN/Inf loss at step %d (%d in last %d steps)",
                self.global_step, len(self._nan_history), self._nan_window,
            )
            if len(self._nan_history) >= self._nan_max_in_window:
                raise RuntimeError(
                    f"Training diverged: {len(self._nan_history)} NaN losses "
                    f"in the last {self._nan_window} steps."
                )
            return None  # skip this step

        for name, value in per_term.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=(name == "total"),
                batch_size=B,
            )
        if self._step_t0 is not None:
            self.log("train/step_time_sec", time.perf_counter() - self._step_t0,
                     on_step=True, on_epoch=False)
        if torch.cuda.is_available():
            self.log(
                "train/gpu_mem_alloc_mb",
                float(torch.cuda.memory_allocated() / (1024 * 1024)),
                on_step=True, on_epoch=False,
            )
        return total

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self.ema is not None:
            self.ema.update()
            self.log(
                "train/ema_decay",
                self.ema.get_current_decay(),
                on_step=True, on_epoch=False,
            )

    # ------------------------------------------------------------------
    # Validation step.
    # ------------------------------------------------------------------

    def _make_ema_call(self) -> Any:
        """Build a model_call(x_t, timestep) closure that runs the EMA shadow."""
        ema_cn = self.ema.ema_model if self.ema is not None else self.controlnet
        ema_cn.eval()

        def model_call(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            B = x_t.shape[0]
            device = x_t.device
            class_labels = self.trunk_config.make_class_labels(B, device)
            spacing = self.trunk_config.make_spacing_tensor(B, device)
            cond = self._val_cond  # set by validation_step before calling sampler
            return self._trunk_forward(
                ema_cn, x_t, timesteps, cond, class_labels, spacing
            )

        return model_call

    def _which_nfes(self, epoch: int) -> list[int]:
        vcfg = self.validation_cfg
        do_sweep = int(vcfg.get("full_sweep_every_epochs", 5)) > 0 and (
            epoch % int(vcfg.get("full_sweep_every_epochs", 5)) == 0
        )
        if do_sweep:
            return [int(n) for n in vcfg.get("sweep_nfes", [1, 2, 5, 10, 50])]
        return [int(vcfg.get("per_epoch_nfe", 5))]

    @torch.inference_mode()
    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        if self.region_resolver is None:
            return  # validation disabled
        masks: RegionMasks = self.region_resolver.resolve(batch)
        z_target = batch["z_t1c"]
        patient_ids = batch.get("patient_id")
        if isinstance(patient_ids, str):
            patient_ids = [patient_ids]

        epoch = int(self.current_epoch)
        nfes = self._which_nfes(epoch)
        per_epoch_nfe = int(self.validation_cfg.get("per_epoch_nfe", 5))
        qual_every = int(self.validation_cfg.get("qualitative_every_epochs", 10))
        do_qual = qual_every > 0 and (epoch % qual_every == 0)

        sampler = EulerSampler(scheduler=self.rflow.scheduler)
        self._val_cond = self.conditioning(batch)
        model_call = self._make_ema_call()

        for nfe in nfes:
            try:
                x0 = torch.randn_like(z_target)
                t_start = time.perf_counter()
                z_pred = sampler.sample(model_call, x0, num_inference_steps=int(nfe))
                t_sample = time.perf_counter() - t_start
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM at NFE=%d epoch=%d — skipping this NFE.", nfe, epoch)
                torch.cuda.empty_cache()
                continue

            self._update_val_accumulator(
                masks=masks, z_pred=z_pred, z_target=z_target, nfe=nfe,
                do_image=(nfe == per_epoch_nfe or len(nfes) > 1),
            )

            self._nfe_timing_buffer.append({
                "epoch": epoch,
                "nfe": nfe,
                "t_total_mean_sec": t_sample / max(1, z_target.shape[0]),
                "t_total_std_sec": 0.0,
                "gpu_mem_peak_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                if torch.cuda.is_available() else 0.0,
                "n_patients_measured": int(z_target.shape[0]),
            })

            if do_qual and patient_ids is not None:
                for b, pid in enumerate(patient_ids):
                    self._qualitative_buffer[(str(pid), int(nfe))] = z_pred[b].detach().cpu().half()

    def _update_val_accumulator(
        self,
        masks: RegionMasks,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        nfe: int,
        do_image: bool,
    ) -> None:
        B = z_pred.shape[0]
        for region in REGION_NAMES:
            mask = masks.get(REGION_TO_RESOLVER_KEY[region])
            key = (int(nfe), region)
            agg = self._val_accumulator.setdefault(key, _new_agg())
            if mask is None:
                agg["n_patients"] = 0
                continue
            mse = self.latent_metrics.mse(z_pred, z_target, mask)
            l1 = self.latent_metrics.l1(z_pred, z_target, mask)
            cos = self.latent_metrics.cosine(z_pred, z_target, mask)
            agg["mse"].extend(mse.detach().cpu().tolist())
            agg["l1"].extend(l1.detach().cpu().tolist())
            agg["cosine"].extend(cos.detach().cpu().tolist())
            agg["n_patients"] = len(agg["mse"])

            # Log per-batch — Lightning aggregates across the validation set
            # into ``trainer.callback_metrics`` which the ModelCheckpoint
            # reads when picking the best epoch.
            self.log(
                f"val/mse_latent_{region}_nfe{nfe}", mse.mean(),
                on_step=False, on_epoch=True, batch_size=B,
            )
            self.log(
                f"val/l1_latent_{region}_nfe{nfe}", l1.mean(),
                on_step=False, on_epoch=True, batch_size=B,
            )
            self.log(
                f"val/cosine_latent_{region}_nfe{nfe}", cos.mean(),
                on_step=False, on_epoch=True, batch_size=B,
            )

            if do_image and self.vae_decoder is not None and self.image_metrics is not None:
                img_pred, img_target = self._decode_pair(z_pred, z_target)
                img_mask = F.interpolate(
                    mask.float(), size=img_pred.shape[-3:], mode="nearest"
                ).bool()
                psnr = self.image_metrics.psnr(img_pred, img_target, img_mask)
                ssim = self.image_metrics.ssim(img_pred, img_target, img_mask)
                agg["psnr"].extend(_safe_tolist(psnr))
                agg["ssim"].extend(_safe_tolist(ssim))
                # Replace NaN entries (empty region) with 0 weight when logging.
                psnr_clean = psnr[torch.isfinite(psnr)] if psnr.numel() else psnr
                ssim_clean = ssim[torch.isfinite(ssim)] if ssim.numel() else ssim
                if psnr_clean.numel() > 0:
                    self.log(
                        f"val/psnr_image_{region}_nfe{nfe}", psnr_clean.mean(),
                        on_step=False, on_epoch=True, batch_size=B,
                    )
                if ssim_clean.numel() > 0:
                    self.log(
                        f"val/ssim_image_{region}_nfe{nfe}", ssim_clean.mean(),
                        on_step=False, on_epoch=True, batch_size=B,
                    )

    def _decode_pair(
        self, z_pred: torch.Tensor, z_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Best-effort decode — exceptions bubble up (validation OOM is caught upstream).
        from vena.model.autoencoder.maisi.preprocessing import DepthPad

        # The H5 stores latents already padded along the depth axis (image-space
        # 155 → 160 → latent 40). For decode-from-latent we ask for an identity
        # un-pad: ``before=after=0`` and ``original_depth == padded_depth ==
        # latent_depth * 4`` (the VAE's 4× compression).
        latent_d = int(z_pred.shape[-1])
        depth = latent_d * 4
        pad = DepthPad(before=0, after=0, original_depth=depth, padded_depth=depth)
        out_pred = self.vae_decoder.decode(z_pred, pad)
        out_target = self.vae_decoder.decode(z_target, pad)
        return out_pred.image, out_target.image

    def on_validation_epoch_end(self) -> None:
        # Collapse list-based aggregators to mean/std. Per-batch ``self.log``
        # calls in ``_update_val_accumulator`` already populated
        # ``trainer.callback_metrics`` for ModelCheckpoint; here we only need
        # to prepare the long-format dict the ValMetricsCSV callback consumes.
        collapsed: dict[tuple[int, str], dict[str, Any]] = {}
        for (nfe, region), agg in self._val_accumulator.items():
            collapsed[(nfe, region)] = _agg_to_stats(agg)
        self._val_accumulator.clear()
        self._val_accumulator.update(collapsed)

    # ------------------------------------------------------------------
    # Checkpoint pathway: RNG state + best metric.
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        rng = checkpoint.get("rng_state")
        if not rng:
            return
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            logger.info("RNG state restored from checkpoint.")
        except Exception as exc:  # noqa: BLE001 — best-effort restore
            logger.warning("RNG restore failed: %s", exc)

    # ------------------------------------------------------------------
    # Optimiser.
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        lr = float(self.optim_cfg.get("lr", 5e-5))
        betas = tuple(self.optim_cfg.get("betas", (0.9, 0.95)))
        weight_decay = float(self.optim_cfg.get("weight_decay", 1e-2))
        warmup_steps = int(self.optim_cfg.get("warmup_steps", 100))
        max_steps = int(self.optim_cfg.get("max_steps", 50_000))
        scheduler_kind = str(self.optim_cfg.get("scheduler", "polynomial")).lower()

        trainable = [p for p in self.controlnet.parameters() if p.requires_grad]
        opt = AdamW(trainable, lr=lr, betas=betas, weight_decay=weight_decay)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            if scheduler_kind == "polynomial":
                remaining = max(0, max_steps - step)
                denom = max(1, max_steps - warmup_steps)
                return max(0.0, remaining / denom)
            return 1.0

        sched = LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


# ----------------------------------------------------------------------
# Aggregator helpers — kept module-level to be picklable for DataLoader workers.
# ----------------------------------------------------------------------


def _new_agg() -> dict[str, Any]:
    return {
        "mse": [], "l1": [], "cosine": [],
        "psnr": [], "ssim": [],
        "n_patients": 0,
    }


def _agg_to_stats(agg: dict[str, Any]) -> dict[str, Any]:
    def mean_std(xs: list[float]) -> tuple[float | None, float | None]:
        if not xs:
            return None, None
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, 0.0
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, math.sqrt(v)

    mse_m, mse_s = mean_std(agg["mse"])
    l1_m, l1_s = mean_std(agg["l1"])
    cos_m, _ = mean_std(agg["cosine"])
    psnr_m, psnr_s = mean_std([x for x in agg["psnr"] if x is not None and not math.isnan(x)])
    ssim_m, ssim_s = mean_std([x for x in agg["ssim"] if x is not None and not math.isnan(x)])
    return {
        "mse_latent_mean": mse_m, "mse_latent_std": mse_s,
        "l1_latent_mean": l1_m, "l1_latent_std": l1_s,
        "cosine_latent_mean": cos_m,
        "psnr_image_mean": psnr_m, "psnr_image_std": psnr_s,
        "ssim_image_mean": ssim_m, "ssim_image_std": ssim_s,
        "n_patients": int(agg["n_patients"]),
    }


def _safe_tolist(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().cpu().tolist()]

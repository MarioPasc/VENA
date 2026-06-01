"""LightningModule wrapping trunk + ControlNet + RFlow + composite loss.

The VAE is always frozen. The trunk is controlled by ``trunk_config.trainable``:

* ``trainable=False`` (canonical frozen-backbone recipe): the optimiser is
  constructed over ``self.controlnet.parameters()`` only and the trunk is held
  as an unregistered property — trunk weights are not written into
  checkpoints and the EMA shadow only tracks the ControlNet.

* ``trainable=True`` (project default since multi-cohort + augmentations work):
  the trunk is unfrozen and joins the same optimiser group as the ControlNet.
  ``self._trunk_module`` is registered as a Lightning submodule so the
  fine-tuned trunk weights round-trip through ``state_dict`` natively (PL 2.x
  restores model weights *after* ``setup()``). A second EMA — ``self.trunk_ema``
  — is built in ``setup()`` and updated in lockstep with the ControlNet EMA so
  sampling uses an EMA-smoothed trunk. Caveat: ``trunk_ema`` is created in
  ``setup()`` (after Lightning's checkpoint restore), so this path is
  **single-shot, not resume-safe** as written. Do not rely on ``run.resume_from``
  for unfrozen runs without first hardening the trunk-EMA restore path.

The training step follows MAISI-v2's ControlNet recipe (see
``.claude/rules/model-coding-standards.md``):

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
from contextlib import nullcontext
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
        from numpy._core.multiarray import _reconstruct

        torch.serialization.add_safe_globals(
            [_reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
        )
    except Exception:
        pass


_register_safe_globals()

from ..controlnet.base import AbstractControlNet
from ..controlnet.conditioning import ConditioningAssembler, ConditioningSpec
from ..controlnet.losses import CompositeLoss, LossInputs, build_loss
from ..controlnet.maisi_controlnet import MaisiControlNet
from ..ema import WarmupEMA
from ..inference import NFETimingProbe, get_sampler
from ..maisi.config import TrunkConfig
from ..maisi.trunk import TrunkHandle, load_trunk
from ..metrics import ImageMetrics, LatentMetrics, RegionMasks, RegionResolver
from ..sampler.rflow import RFlowEngine

logger = logging.getLogger(__name__)


REGION_NAMES: tuple[str, ...] = ("full", "wt", "bg", "vessel")
REGION_TO_RESOLVER_KEY: dict[str, str] = {
    "full": "brain",  # full-brain mask
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
        self.save_hyperparameters(ignore=["trunk_config", "region_resolver", "vae_decoder"])

        self.trunk_config = trunk_config
        self.stage = stage
        self.perturb_keys: set[str] = set(perturb_keys or ()) if perturb_keys else {"wt"}

        self._trunk_handle: TrunkHandle | None = None
        # Registered alias for the live trunk, set in setup() only when the trunk
        # is trainable. Registration puts the fine-tuned trunk weights into the
        # Lightning state_dict so they are saved and restored natively (PL 2.x
        # restores model weights *after* setup()). Frozen trunk stays unregistered
        # so frozen checkpoints are not bloated with 72 M immutable params.
        self._trunk_module: torch.nn.Module | None = None
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
        # Trunk EMA only exists in the unfrozen-trunk ablation; it is built in
        # setup() once the trunk is loaded (the trunk does not exist in
        # __init__). This path is single-shot (not resume-safe), by design.
        self.trunk_ema: WarmupEMA | None = None

        # Validation/region wiring.
        self.region_resolver = region_resolver
        self.validation_cfg: dict[str, Any] = validation_cfg or {}
        self.vae_decoder = vae_decoder
        self.latent_metrics = LatentMetrics()
        self.image_metrics: ImageMetrics | None = None  # built lazily if vae_decoder set

        # Buffers consumed by callbacks.
        self._val_accumulator: dict[tuple[int, str], dict[str, Any]] = {}
        self._qualitative_buffer: dict[tuple[str, int], torch.Tensor] = {}
        # Per-epoch NFE timing accumulator, keyed by nfe. Each value collects
        # per-batch lists so the callback can emit one aggregated row per
        # (epoch, nfe) instead of one row per validation batch.
        self._nfe_timing_accum: dict[int, dict[str, Any]] = {}

        # NaN guard counters.
        nt = nan_tolerance or {}
        self._nan_max_in_window = int(nt.get("max_in_window", 10))
        self._nan_window = int(nt.get("window_steps", 1000))
        self._nan_history: list[int] = []  # step indices where NaN occurred

        # Step-time tracking (logged per training step).
        self._step_t0: float | None = None
        # Last optimiser step at which the EMA updated (grad-accum guard).
        # 0 = "no optimiser step yet"; ``global_step`` advances past it only
        # once a real optimiser step completes.
        self._last_ema_step: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if self._trunk_handle is None:
            self._setup_trunk_and_controlnet()
        # Move EMA shadow to the same device as the live model.
        self.ema = self.ema.to(self.device)
        # Unfrozen-trunk ablation: a second EMA over the trunk so that sampling
        # (validation + exhaustive job) uses EMA-smoothed trunk weights exactly
        # as it uses the EMA ControlNet. Built here because the trunk does not
        # exist until ``_setup_trunk_and_controlnet`` has run.
        if self.trunk_config.trainable and self.trunk_ema is None:
            self.trunk_ema = WarmupEMA(self.trunk, **self.ema_cfg).to(self.device)
            logger.info("Trunk EMA shadow created (unfrozen-trunk ablation).")
        if self.image_metrics is None and self.vae_decoder is not None:
            self.image_metrics = ImageMetrics()

    def _setup_trunk_and_controlnet(self) -> None:
        ckpt = Path(self.trunk_config.checkpoint)
        arch_json = Path(self.trunk_config.arch_json) if self.trunk_config.arch_json else None
        self._trunk_handle = load_trunk(
            checkpoint_path=ckpt,
            device=self.device,
            arch_config=arch_json,
            arch_overrides=self.trunk_config.arch_overrides or None,
            trainable=self.trunk_config.trainable,
        )
        if self.trunk_config.trainable:
            # Register the trunk so its weights are checkpointed and restored
            # natively. On resume, setup() reloads the *original* MAISI trunk here
            # (harmless), then Lightning's post-setup state_dict restore overwrites
            # it (and trunk_ema + optimiser state) with the fine-tuned values.
            self._trunk_module = self._trunk_handle.model
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
        probe: NFETimingProbe | None = None,
        trunk: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # ``trunk`` defaults to the live trunk (training path). The EMA-call
        # closure passes the EMA trunk shadow so sampling uses smoothed weights.
        trunk_model = trunk if trunk is not None else self.trunk
        x_t_p, pad = self._pad_to_multiple(x_t, multiple=8)
        cond_p, _ = self._pad_to_multiple(cond, multiple=8)
        cn_ctx = probe.section("controlnet") if probe is not None else nullcontext()
        with cn_ctx:
            down_res, mid_res = controlnet(
                x=x_t_p,
                timesteps=timesteps,
                controlnet_cond=cond_p,
                class_labels=class_labels,
            )
        trunk_ctx = probe.section("trunk") if probe is not None else nullcontext()
        with trunk_ctx:
            v_p = trunk_model(
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
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor | None:
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
                x_t,
                timesteps,
                cond_perturb,
                class_labels,
                spacing,
            )

        m_wt = batch.get("m_wt")
        m_bg = (
            _bg_from_wt(m_wt)
            if (m_wt is not None and self.composite.requires_perturbed_pass)
            else None
        )
        inputs = LossInputs(
            x_clean=x1,
            noise=x0,
            x_t=x_t,
            timesteps=timesteps,
            u_target=u_target,
            v_orig=v_orig,
            v_perturb=v_perturb,
            m_wt=m_wt,
            m_bg=m_bg,
        )
        total_steps = self._estimated_total_steps()
        total, per_term = self.composite(
            inputs,
            global_step=int(self.global_step),
            total_steps=total_steps,
        )
        # Per-cohort CFM breakdown (P1.2). The multi-cohort dataset attaches a
        # ``cohort`` string per sample; the DataLoader collates strings into a
        # list. When all samples are from one cohort the .mean() across that
        # cohort equals the global cfm; otherwise the per-cohort values diverge
        # and reveal cohort-imbalanced drift.
        cohort_tags = batch.get("cohort")
        if cohort_tags is not None and v_orig.shape[0] > 1:
            # CFM is MSE between v_orig and u_target; recompute per-sample then
            # group by cohort (cheap: B≤8, mean is sub-microsecond).
            per_sample = (v_orig.detach() - u_target).pow(2).flatten(1).mean(dim=1)
            cohort_groups: dict[str, list[float]] = {}
            for i, tag in enumerate(cohort_tags):
                cohort_groups.setdefault(str(tag), []).append(float(per_sample[i].item()))
            for tag, vals in cohort_groups.items():
                # Sanitise cohort name for CSV column compatibility.
                safe = tag.replace("/", "_").replace(" ", "_")
                self.log(
                    f"train/cfm_cohort_{safe}",
                    sum(vals) / len(vals),
                    on_step=True,
                    on_epoch=False,
                    batch_size=len(vals),
                )

        # NaN guard.
        if not torch.isfinite(total):
            self._nan_history.append(int(self.global_step))
            self._nan_history = [
                s for s in self._nan_history if self.global_step - s <= self._nan_window
            ]
            logger.error(
                "NaN/Inf loss at step %d (%d in last %d steps)",
                self.global_step,
                len(self._nan_history),
                self._nan_window,
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
        # Epoch-aggregated training loss under a distinct key — the
        # checkpoint monitor (ema_best) selects on this when in-process
        # validation is offloaded to the async second-GPU job. Distinct name so
        # the per-step ``train/total`` key the train CSV reads is not renamed.
        self.log("train/total_epoch", total, on_step=False, on_epoch=True, batch_size=B)
        # Sanity on the timestep sampler (should hover near T/2 for uniform).
        self.log("train/t_mean", timesteps.float().mean(), on_step=True, on_epoch=False)
        if self._step_t0 is not None:
            step_time = time.perf_counter() - self._step_t0
            self.log("train/step_time_sec", step_time, on_step=True, on_epoch=False)
            self.log(
                "train/samples_per_sec",
                float(B) / max(step_time, 1e-9),
                on_step=True,
                on_epoch=False,
            )
        if torch.cuda.is_available():
            # Peak (not current) allocation gives OOM headroom; reset in
            # ``on_train_batch_start`` so this reflects the step just executed.
            self.log(
                "train/gpu_mem_peak_mb",
                float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                on_step=True,
                on_epoch=False,
            )
        return total

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.ema is None:
            return
        # ``on_train_batch_end`` fires once per *micro-batch*. With gradient
        # accumulation (``accumulate_grad_batches > 1``) the optimizer steps
        # only every N micro-batches, so the EMA must update once per optimiser
        # step — not per micro-batch — or the shadow decays N× too fast.
        # ``trainer.global_step`` increments only on optimiser steps, so we gate
        # on it *advancing* — this also skips the pre-first-step accumulation
        # micro-batches where ``global_step`` is still 0.
        step = int(self.trainer.global_step)
        if step <= self._last_ema_step:
            return
        self._last_ema_step = step
        self.ema.update()
        if self.trunk_ema is not None:
            # Same once-per-optimiser-step gate as the ControlNet EMA above.
            self.trunk_ema.update()
        self.log(
            "train/ema_decay",
            self.ema.get_current_decay(),
            on_step=True,
            on_epoch=False,
        )

    def _estimated_total_steps(self) -> int | None:
        """Total optimiser-step budget for this run, used by weight schedules.

        Prefers Lightning's ``trainer.estimated_stepping_batches`` (available
        once ``trainer.fit`` has computed the dataloader size), falling back to
        ``optim_cfg["max_steps"]`` (read from the YAML's ``training.total_steps``
        in the engine) when the trainer is not attached yet. Returns ``None``
        when neither is known so schedules can no-op.
        """
        if getattr(self, "trainer", None) is not None:
            try:
                est = int(self.trainer.estimated_stepping_batches)
                if est > 0:
                    return est
            except (AttributeError, ValueError, TypeError):
                pass
        ms = self.optim_cfg.get("max_steps")
        try:
            ms_int = int(ms) if ms is not None else None
            return ms_int if ms_int and ms_int > 0 else None
        except (TypeError, ValueError):
            return None

    def _trainable_grad_norm(self) -> torch.Tensor:
        """Global L2 norm over all optimised parameters.

        ControlNet always; plus the trunk when ``trunk_config.trainable`` (the
        unfrozen-trunk ablation), so the logged norm matches the parameter set
        the gradient clip actually acts on.
        """
        sq_sum = torch.zeros((), device=self.device)
        params = list(self.controlnet.parameters())
        if self.trunk_config.trainable:
            params += list(self.trunk.parameters())
        for p in params:
            if p.grad is not None:
                sq_sum = sq_sum + p.grad.detach().float().pow(2).sum()
        return sq_sum.sqrt()

    def _trunk_grad_norm(self) -> torch.Tensor:
        """Global L2 norm over the trunk parameters only (unfrozen-trunk run).

        Returned independently so the train CSV can monitor the trunk's
        gradient magnitude alongside the combined-norm and detect the case where
        the trunk explodes while the ControlNet stays quiet (or vice versa).
        Callers gate on ``self.trunk_config.trainable``.
        """
        sq_sum = torch.zeros((), device=self.device)
        for p in self.trunk.parameters():
            if p.grad is not None:
                sq_sum = sq_sum + p.grad.detach().float().pow(2).sum()
        return sq_sum.sqrt()

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip ControlNet gradients and log pre/post-clip norms.

        Lightning calls this once per optimiser step (grad-accum-safe), after
        ``backward`` and before ``optimizer.step``. We measure the norm before
        and after the clip so the logs show both the raw gradient magnitude
        (stability signal) and the effective post-clip norm, plus whether the
        clip was active this step.
        """
        pre = self._trainable_grad_norm()
        trunk_pre: torch.Tensor | None = (
            self._trunk_grad_norm() if self.trunk_config.trainable else None
        )
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        post = self._trainable_grad_norm()
        trunk_post: torch.Tensor | None = (
            self._trunk_grad_norm() if self.trunk_config.trainable else None
        )
        # ``grad_norm_cn_*`` is misnamed historically — it is the combined
        # ControlNet + (unfrozen) trunk norm. Kept as-is for CSV back-compat with
        # earlier runs. The trunk-only keys below are the actual decomposition.
        self.log("train/grad_norm_cn_preclip", pre, on_step=True, on_epoch=False)
        self.log("train/grad_norm_cn_postclip", post, on_step=True, on_epoch=False)
        if trunk_pre is not None and trunk_post is not None:
            self.log("train/grad_norm_trunk_preclip", trunk_pre, on_step=True, on_epoch=False)
            self.log("train/grad_norm_trunk_postclip", trunk_post, on_step=True, on_epoch=False)
        if gradient_clip_val:
            self.log(
                "train/grad_clip_active",
                (pre > float(gradient_clip_val)).float(),
                on_step=True,
                on_epoch=False,
            )

    # ------------------------------------------------------------------
    # Validation step.
    # ------------------------------------------------------------------

    def _make_ema_call(self, probe: NFETimingProbe | None = None) -> Any:
        """Build a model_call(x_t, timestep) closure that runs the EMA shadow.

        When ``probe`` is given, the controlnet and trunk forwards inside each
        sampler step are wrapped in CUDA-synchronised timing sections so the
        per-component NFE timing can be reported.
        """
        ema_cn = self.ema.ema_model if self.ema is not None else self.controlnet
        ema_cn.eval()
        ema_trunk = self._ema_trunk()

        def model_call(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            B = x_t.shape[0]
            device = x_t.device
            class_labels = self.trunk_config.make_class_labels(B, device)
            spacing = self.trunk_config.make_spacing_tensor(B, device)
            cond = self._val_cond  # set by validation_step before calling sampler
            return self._trunk_forward(
                ema_cn, x_t, timesteps, cond, class_labels, spacing, probe=probe, trunk=ema_trunk
            )

        return model_call

    def _ema_trunk(self) -> torch.nn.Module:
        """Trunk to sample with: EMA shadow when fine-tuning, else the live trunk.

        In the frozen-trunk default this returns ``self.trunk`` unchanged, so the
        frozen sampling path is identical to before.
        """
        if self.trunk_config.trainable and self.trunk_ema is not None:
            shadow = self.trunk_ema.ema_model
            shadow.eval()
            return shadow
        return self.trunk

    def compute_val_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the validation conditioning tensor for ``batch`` and stash it.

        Called from ``validation_step`` and from the external exhaustive-val
        engine. The result is also written to ``self._val_cond`` so the closure
        returned by :meth:`_make_ema_call` reads the same conditioning across
        every NFE in the sweep.
        """
        self._val_cond = self.conditioning(batch)
        return self._val_cond

    def _which_nfes(self, epoch: int) -> list[int]:
        vcfg = self.validation_cfg
        do_sweep = int(vcfg.get("full_sweep_every_epochs", 5)) > 0 and (
            epoch % int(vcfg.get("full_sweep_every_epochs", 5)) == 0
        )
        if do_sweep:
            return [int(n) for n in vcfg.get("sweep_nfes", [1, 2, 5, 10, 50])]
        return [int(vcfg.get("per_epoch_nfe", 5))]

    def _do_image_metrics(self, epoch: int) -> bool:
        """Whether to decode to image space and compute PSNR/SSIM this epoch.

        Image-space metrics are expensive (one VAE decode per patient, ~2.5 s)
        and the small-region SSIM is noisy, so they run on a slow cadence
        (``validation.image_metrics_every_epochs``) rather than every epoch, and
        only at the canonical ``per_epoch_nfe``. ``0`` disables them entirely.
        """
        if not self.validation_cfg.get("image_metrics", True):
            return False
        if self.vae_decoder is None or self.image_metrics is None:
            return False
        every = int(self.validation_cfg.get("image_metrics_every_epochs", 0))
        return every > 0 and (epoch % every == 0)

    @torch.inference_mode()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
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
        do_image_epoch = self._do_image_metrics(epoch)

        sampler = get_sampler(self.validation_cfg.get("integrator", "euler"))(
            scheduler=self.rflow.scheduler
        )
        # Materialise the conditioning once per batch; the EMA closure
        # constructed by ``_make_ema_call`` reads ``self._val_cond`` so the
        # same conditioning is reused across NFE values.
        self._val_cond = self.compute_val_conditioning(batch)
        B = int(z_target.shape[0])

        for nfe in nfes:
            probe = NFETimingProbe()
            model_call = self._make_ema_call(probe=probe)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                x0 = torch.randn_like(z_target)
                t_start = time.perf_counter()
                z_pred = sampler.sample(model_call, x0, num_inference_steps=int(nfe))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_sample = time.perf_counter() - t_start
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM at NFE=%d epoch=%d — skipping this NFE.", nfe, epoch)
                torch.cuda.empty_cache()
                continue

            decode_sec = self._update_val_accumulator(
                masks=masks,
                z_pred=z_pred,
                z_target=z_target,
                nfe=nfe,
                do_image=(do_image_epoch and int(nfe) == per_epoch_nfe),
            )

            # Per-component timings: drop the first sampler step (CUDA warm-up).
            comp = probe.aggregate(drop_first=int(nfe) > 1)
            self._accumulate_nfe_timing(
                nfe=int(nfe),
                t_total_per_patient=t_sample / max(1, B),
                t_trunk=comp.get("trunk", {}).get("mean", float("nan")),
                t_controlnet=comp.get("controlnet", {}).get("mean", float("nan")),
                t_decode_per_patient=(decode_sec / max(1, B)) if decode_sec is not None else None,
                gpu_mem_peak_mb=(
                    float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                    if torch.cuda.is_available()
                    else 0.0
                ),
                n_patients=B,
            )

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
    ) -> float | None:
        """Update region metrics for one (nfe, batch).

        Returns
        -------
        float | None
            Wall-clock seconds to decode the predicted volume(s) through the
            VAE (one decode for the whole batch), or ``None`` when image
            metrics are disabled / no decoder is available. Decoding happens
            once here and the resulting images are reused across regions.
        """
        B = z_pred.shape[0]
        decode_sec: float | None = None
        img_pred: torch.Tensor | None = None
        img_target: torch.Tensor | None = None
        if do_image and self.vae_decoder is not None and self.image_metrics is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            img_pred, img_target = self._decode_pair(z_pred, z_target)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_sec = time.perf_counter() - t0

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
                f"val/mse_latent_{region}_nfe{nfe}",
                mse.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                f"val/l1_latent_{region}_nfe{nfe}",
                l1.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )
            self.log(
                f"val/cosine_latent_{region}_nfe{nfe}",
                cos.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=B,
            )

            if img_pred is not None and img_target is not None:
                img_mask = F.interpolate(
                    mask.float(), size=img_pred.shape[-3:], mode="nearest"
                ).bool()
                psnr = self.image_metrics.psnr(img_pred, img_target, img_mask)
                ssim = self.image_metrics.ssim(img_pred, img_target, img_mask)
                agg["psnr"].extend(_safe_tolist(psnr))
                agg["ssim"].extend(_safe_tolist(ssim))
                agg["n_image_patients"] = len(agg["psnr"])
                # Replace NaN entries (empty region) with 0 weight when logging.
                psnr_clean = psnr[torch.isfinite(psnr)] if psnr.numel() else psnr
                ssim_clean = ssim[torch.isfinite(ssim)] if ssim.numel() else ssim
                if psnr_clean.numel() > 0:
                    self.log(
                        f"val/psnr_image_{region}_nfe{nfe}",
                        psnr_clean.mean(),
                        on_step=False,
                        on_epoch=True,
                        batch_size=B,
                    )
                if ssim_clean.numel() > 0:
                    self.log(
                        f"val/ssim_image_{region}_nfe{nfe}",
                        ssim_clean.mean(),
                        on_step=False,
                        on_epoch=True,
                        batch_size=B,
                    )

        return decode_sec

    def _decode_pair(
        self, z_pred: torch.Tensor, z_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Best-effort decode — exceptions bubble up (validation OOM is caught upstream).
        from vena.common.decode import decode_depth_identity

        out_pred = decode_depth_identity(self.vae_decoder, z_pred)
        out_target = decode_depth_identity(self.vae_decoder, z_target)
        return out_pred.image, out_target.image

    # ------------------------------------------------------------------
    # NFE timing accumulation (per-component, aggregated per epoch).
    # ------------------------------------------------------------------

    def _accumulate_nfe_timing(
        self,
        nfe: int,
        t_total_per_patient: float,
        t_trunk: float,
        t_controlnet: float,
        t_decode_per_patient: float | None,
        gpu_mem_peak_mb: float,
        n_patients: int,
    ) -> None:
        acc = self._nfe_timing_accum.setdefault(nfe, _new_timing_agg())
        acc["t_total"].append(float(t_total_per_patient))
        acc["t_trunk"].append(float(t_trunk))
        acc["t_controlnet"].append(float(t_controlnet))
        if t_decode_per_patient is not None:
            acc["t_decode"].append(float(t_decode_per_patient))
        acc["gpu_mem_peak_mb"] = max(acc["gpu_mem_peak_mb"], float(gpu_mem_peak_mb))
        acc["n_patients"] += int(n_patients)

    def collapse_nfe_timing(self) -> list[dict[str, Any]]:
        """Aggregate the per-epoch NFE timing accumulator to one row per nfe.

        Pure read (no mutation), mirroring :meth:`collapse_val_metrics`. The
        ``NFETimingCSV`` callback consumes this on ``on_validation_epoch_end``;
        the accumulator itself is cleared in this module's later-firing
        ``on_validation_epoch_end``.

        Returns
        -------
        list[dict[str, Any]]
            One dict per nfe with the columns the CSV expects.
        """
        rows: list[dict[str, Any]] = []
        for nfe in sorted(self._nfe_timing_accum):
            acc = self._nfe_timing_accum[nfe]
            rows.append(
                {
                    "nfe": int(nfe),
                    "t_trunk_mean_sec": _finite_mean(acc["t_trunk"]),
                    "t_controlnet_mean_sec": _finite_mean(acc["t_controlnet"]),
                    "t_decode_sec": _finite_mean(acc["t_decode"]),
                    "t_total_mean_sec": _finite_mean(acc["t_total"]),
                    "t_total_std_sec": _finite_std(acc["t_total"]),
                    "gpu_mem_peak_mb": acc["gpu_mem_peak_mb"],
                    "n_patients_measured": int(acc["n_patients"]),
                }
            )
        return rows

    def collapse_val_metrics(self) -> dict[tuple[int, str], dict[str, Any]]:
        """Collapse the raw per-region accumulator to mean/std stats.

        Pure read: does not mutate ``self._val_accumulator``. The
        ``ValMetricsCSV`` callback calls this on ``on_validation_epoch_end``.
        It must be a separate method (not done in-place in this module's own
        ``on_validation_epoch_end``) because Lightning fires
        ``Callback.on_validation_epoch_end`` *before*
        ``LightningModule.on_validation_epoch_end`` — so an in-place collapse
        here would run too late and the callback would read raw lists.

        Returns
        -------
        dict[tuple[int, str], dict[str, Any]]
            Mapping ``(nfe, region)`` to the collapsed stat dict produced by
            :func:`_agg_to_stats` (``*_mean`` / ``*_std`` / ``n_patients``).
        """
        return {
            (nfe, region): _agg_to_stats(agg)
            for (nfe, region), agg in self._val_accumulator.items()
        }

    def on_validation_epoch_end(self) -> None:
        # Fires *after* every callback's ``on_validation_epoch_end`` (Lightning
        # calls callback hooks before the module hook). By this point the
        # ValMetricsCSV callback has already consumed the accumulator via
        # ``collapse_val_metrics``; clear it so the next epoch starts fresh.
        # Per-batch ``self.log`` calls in ``_update_val_accumulator`` already
        # populated ``trainer.callback_metrics`` for ModelCheckpoint.
        self._val_accumulator.clear()
        self._nfe_timing_accum.clear()

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
        # When the trunk is trainable it is a registered submodule
        # (``_trunk_module``) and its EMA (``trunk_ema``) is registered too, so
        # both are already in ``checkpoint["state_dict"]`` and restored natively
        # on resume — no custom payload needed.

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
        except Exception as exc:
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
        if self.trunk_config.trainable:
            # Unfrozen-trunk ablation: fine-tune the trunk jointly with the
            # ControlNet under a single param group (matches TumorFlow's recipe).
            trunk_params = [p for p in self.trunk.parameters() if p.requires_grad]
            trainable += trunk_params
            logger.info(
                "configure_optimizers: trunk UNFROZEN — optimising %d controlnet + %d trunk tensors",
                len(trainable) - len(trunk_params),
                len(trunk_params),
            )
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


def _bg_from_wt(m_wt: torch.Tensor) -> torch.Tensor:
    """Dilated-complement background mask in latent space.

    ``m_bg = 1 - dilate3(m_wt)`` where ``dilate3`` is a 3×3×3 max-pool — the
    same primitive used by ``RegionMasks._dilate_wt`` for the validation-time WT
    dilation. The complement spans every voxel that is *not* within one latent
    voxel of the tumour, matching proposal §5.3 step 1.

    Parameters
    ----------
    m_wt : Tensor
        Binary whole-tumour mask in latent space, shape ``(B, 1, h, w, d)``.

    Returns
    -------
    Tensor
        Background mask of the same shape, valued in ``{0, 1}``.
    """
    m = m_wt.to(dtype=torch.float32)
    dilated = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
    return (1.0 - dilated).clamp_(0.0, 1.0)


def _new_agg() -> dict[str, Any]:
    return {
        "mse": [],
        "l1": [],
        "cosine": [],
        "psnr": [],
        "ssim": [],
        "n_patients": 0,
        "n_image_patients": 0,
    }


def _new_timing_agg() -> dict[str, Any]:
    return {
        "t_total": [],
        "t_trunk": [],
        "t_controlnet": [],
        "t_decode": [],
        "gpu_mem_peak_mb": 0.0,
        "n_patients": 0,
    }


def _finite_mean(xs: list[float]) -> float | None:
    """Mean over the finite (non-NaN) entries; ``None`` if no finite samples."""
    finite = [x for x in xs if not math.isnan(x)]
    return sum(finite) / len(finite) if finite else None


def _finite_std(xs: list[float]) -> float | None:
    """Sample stddev (Bessel) over the finite entries.

    Returns ``0.0`` for a single finite value and ``None`` when no finite
    samples exist. Matches the legacy nested ``_std`` semantics.
    """
    finite = [x for x in xs if not math.isnan(x)]
    if len(finite) < 2:
        return 0.0 if finite else None
    m = sum(finite) / len(finite)
    return math.sqrt(sum((x - m) ** 2 for x in finite) / (len(finite) - 1))


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
        "mse_latent_mean": mse_m,
        "mse_latent_std": mse_s,
        "l1_latent_mean": l1_m,
        "l1_latent_std": l1_s,
        "cosine_latent_mean": cos_m,
        "psnr_image_mean": psnr_m,
        "psnr_image_std": psnr_s,
        "ssim_image_mean": ssim_m,
        "ssim_image_std": ssim_s,
        "n_patients": int(agg["n_patients"]),
        "n_image_patients": int(agg.get("n_image_patients", 0)),
    }


def _safe_tolist(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().cpu().tolist()]

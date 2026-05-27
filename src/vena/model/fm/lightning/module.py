"""LightningModule wrapping trunk + ControlNet + RFlow + composite loss.

Trunk is frozen; only ControlNet parameters are trained. The optimiser is
constructed over ``self.controlnet.parameters()`` only.

The training step follows MAISI-v2's ControlNet recipe (training tutorial,
NV-Generate-CTMR/scripts/train_controlnet.py) with the splice:

    down_residuals, mid_residual = controlnet(x_t, t, c_orig, class_labels)
    v = trunk(
        x_t, t,
        class_labels=class_labels,
        spacing_tensor=spacing,
        down_block_additional_residuals=down_residuals,
        mid_block_additional_residual=mid_residual,
    )
    loss = composite(LossInputs(..., v_orig=v, v_perturb=optional))

For S1 the composite contains only :class:`CFMLoss` and
``requires_perturbed_pass=False``, so the second ControlNet forward is skipped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..controlnet.base import AbstractControlNet
from ..controlnet.conditioning import ConditioningAssembler, ConditioningSpec
from ..controlnet.losses import CompositeLoss, LossInputs, build_loss
from ..controlnet.maisi_controlnet import MaisiControlNet
from ..maisi.config import TrunkConfig
from ..maisi.trunk import TrunkHandle, load_trunk
from ..sampler.rflow import RFlowEngine

logger = logging.getLogger(__name__)


class FMLightningModule(pl.LightningModule):
    """End-to-end FM training step (ControlNet only).

    Parameters
    ----------
    trunk_config : TrunkConfig
        Frozen-trunk loading config.
    conditioning_specs : list
        Ordered list of conditioning input descriptors (strings or
        :class:`ConditioningSpec`).
    stage : str
        Curriculum stage (``"S1"`` for the smoke).
    loss_cfg : dict
        Loss-block config consumed by
        :func:`vena.model.fm.controlnet.losses.build_loss`.
    perturb_keys : set[str] | None
        Set of conditioning spec keys to perturb (zero) on the second
        ControlNet pass. ``{"wt"}`` is the default for S2/S3; ignored for S1
        (the composite reports ``requires_perturbed_pass=False``).
    controlnet_arch_overrides : dict | None
        Forwarded to :class:`MaisiControlNet`.
    optim_cfg : dict | None
        Optimiser hyperparameters: ``lr``, ``betas``, ``weight_decay``,
        ``warmup_steps``, ``max_steps``, ``scheduler``.
    rflow_cfg : dict | None
        :class:`RFlowEngine` kwargs.
    """

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
    ) -> None:
        super().__init__()
        # Saved hparams for Lightning's checkpoint plumbing (strings only —
        # complex objects are reconstructed from these on resume).
        self.save_hyperparameters(ignore=["trunk_config"])

        self.trunk_config = trunk_config
        self.stage = stage
        self.perturb_keys: set[str] = set(perturb_keys or ()) if perturb_keys else {"wt"}

        # Trunk: built on `setup` so the device is known and the trunk lands
        # on the right GPU without an extra .to() shuffle.
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

    # ------------------------------------------------------------------
    # Setup: load the frozen trunk on the same device Lightning chose.
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if self._trunk_handle is not None:
            return
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
        # Warm-start the ControlNet encoder from the trunk *before* the first
        # forward, then zero the output projections so step 0 reproduces the
        # frozen trunk.
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
    # Forward pass.
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor, multiple: int = 8
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Symmetric-on-right pad to make every spatial dim divisible by ``multiple``.

        The MAISI trunk's 4-level U-Net performs three stride-2 downsamples,
        so spatial dims must be divisible by ``2**3 = 8`` for the
        skip-connection concatenations to line up on the way back up. UCSF-PDGM
        latents are ``(60, 60, 40)`` — 40 is fine, 60 is not. We pad on the
        right only to keep the un-pad arithmetic a single slice per axis.

        Returns
        -------
        padded : Tensor
            ``(B, C, H', W', D')`` with every spatial dim a multiple of
            ``multiple``.
        pad : tuple[int, int, int]
            Per-axis right-padding ``(pad_H, pad_W, pad_D)``; subtract from
            the corresponding spatial dim to recover the original tensor.
        """
        sizes = x.shape[-3:]
        pad_h = (multiple - sizes[0] % multiple) % multiple
        pad_w = (multiple - sizes[1] % multiple) % multiple
        pad_d = (multiple - sizes[2] % multiple) % multiple
        # ``F.pad`` takes pairs in reverse-axis order: (D_left, D_right, W_left, W_right, H_left, H_right).
        if pad_h == 0 and pad_w == 0 and pad_d == 0:
            return x, (0, 0, 0)
        padded = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        return padded, (pad_h, pad_w, pad_d)

    @staticmethod
    def _unpad(x: torch.Tensor, pad: tuple[int, int, int]) -> torch.Tensor:
        pad_h, pad_w, pad_d = pad
        if pad_h == 0 and pad_w == 0 and pad_d == 0:
            return x
        return x[
            ...,
            : x.shape[-3] - pad_h,
            : x.shape[-2] - pad_w,
            : x.shape[-1] - pad_d,
        ]

    def _trunk_forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        class_labels: torch.Tensor,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        # The trunk's three down/up samples require dims divisible by 8.
        x_t_p, pad = self._pad_to_multiple(x_t, multiple=8)
        cond_p, _ = self._pad_to_multiple(cond, multiple=8)
        down_res, mid_res = self.controlnet(
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

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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
        v_orig = self._trunk_forward(x_t, timesteps, cond_orig, class_labels, spacing)

        v_perturb: torch.Tensor | None = None
        if self.composite.requires_perturbed_pass:
            cond_perturb = self.conditioning(batch, perturb_keys=self.perturb_keys)
            v_perturb = self._trunk_forward(
                x_t, timesteps, cond_perturb, class_labels, spacing
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

        for name, value in per_term.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=(name == "total"),
                batch_size=B,
            )
        return total

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
            return 1.0  # constant

        sched = LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

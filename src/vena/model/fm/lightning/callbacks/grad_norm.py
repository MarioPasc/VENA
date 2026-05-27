"""ControlNet-only gradient-norm logger.

Lightning has built-in grad-norm logging via ``track_grad_norm``, but it
aggregates over *all* trainable parameters and runs pre-clip. We want the
post-clip norm restricted to the ControlNet parameters — the only part of
VENA that learns.

Implementation: ``on_before_optimizer_step`` runs after Lightning has applied
its gradient clipping, so the recorded norm is the post-clip value. We log
``train/grad_norm_cn`` as a scalar.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch


class GradNormLogger(pl.Callback):
    """Logs the post-clip global L2-norm of ControlNet gradients."""

    def __init__(self, attr_name: str = "controlnet", log_key: str = "train/grad_norm_cn") -> None:
        super().__init__()
        self.attr_name = attr_name
        self.log_key = log_key

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        target = getattr(pl_module, self.attr_name, None)
        if target is None:
            return
        sq_sum = torch.zeros((), device=pl_module.device)
        for p in target.parameters():
            if p.grad is not None:
                sq_sum = sq_sum + p.grad.detach().float().pow(2).sum()
        norm = sq_sum.sqrt()
        pl_module.log(self.log_key, norm, on_step=True, on_epoch=False, prog_bar=False)

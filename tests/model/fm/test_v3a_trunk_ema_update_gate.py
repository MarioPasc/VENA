"""Regression: ``on_train_batch_end`` must update ``trunk_ema`` even when ``self.ema is None``.

The S1 v3 Variant A configuration has no ControlNet, so ``self.ema is None``,
but the trunk is fine-tuned and ``self.trunk_ema`` carries the smoothed
sampling weights. A 2026-06-22 bug short-circuited ``on_train_batch_end`` on
``self.ema is None``, freezing the trunk EMA shadow at its init state and
causing every exhaustive_val sample to be unconditional MAISI noise. This
test guards the gate by exercising the update path with a stub module.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vena.model.fm.ema import WarmupEMA
from vena.model.fm.lightning.module import FMLightningModule


@pytest.mark.unit
def test_trunk_ema_updates_when_cn_ema_absent() -> None:
    """v3a path: ``self.ema is None`` must NOT short-circuit ``trunk_ema.update()``."""
    trunk = nn.Linear(4, 4)
    trunk_ema = WarmupEMA(trunk, decay=0.5, inv_gamma=0.001, power=1.0)

    # Pull the shadow off the live trunk so any subsequent update is observable.
    with torch.no_grad():
        for p in trunk.parameters():
            p.fill_(1.0)
    shadow_before = next(trunk_ema.ema_model.parameters()).detach().clone()

    # Minimal duck-typed surface used by on_train_batch_end. SimpleNamespace
    # avoids the nn.Module __init__ contract that FMLightningModule inherits.
    stub = SimpleNamespace(
        ema=None,
        trunk_ema=trunk_ema,
        trainer=SimpleNamespace(global_step=1),
        _last_ema_step=0,
        log=lambda *a, **k: None,
    )

    FMLightningModule.on_train_batch_end(stub, outputs=None, batch=None, batch_idx=0)

    shadow_after = next(trunk_ema.ema_model.parameters()).detach().clone()
    # WarmupEMA pulled the shadow toward the live (now 1.0-filled) trunk.
    assert not torch.allclose(shadow_before, shadow_after), (
        "trunk_ema shadow did not update — on_train_batch_end short-circuited "
        "on self.ema is None and skipped trunk_ema.update()."
    )
    assert stub._last_ema_step == 1


@pytest.mark.unit
def test_on_train_batch_end_returns_when_both_emas_absent() -> None:
    """Frozen-trunk + no-CN edge case: both EMAs absent ⇒ early return."""
    stub = SimpleNamespace(
        ema=None,
        trunk_ema=None,
        trainer=SimpleNamespace(global_step=1),
        _last_ema_step=0,
        log=lambda *a, **k: None,
    )
    FMLightningModule.on_train_batch_end(stub, outputs=None, batch=None, batch_idx=0)
    # _last_ema_step must NOT have advanced — confirms the early return fired.
    assert stub._last_ema_step == 0

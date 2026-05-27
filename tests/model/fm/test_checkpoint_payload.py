"""Test the LightningModule's on_save / on_load checkpoint RNG payload."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from unittest.mock import patch

from vena.model.fm.maisi.config import TrunkConfig


@pytest.mark.unit
def test_rng_state_roundtrip_through_checkpoint_payload(monkeypatch) -> None:
    """``on_save_checkpoint`` embeds RNG state; ``on_load_checkpoint`` restores it."""
    # Use an isolated stub by importing the module under patched trunk-load
    # so we avoid the real MAISI checkpoint dependency.
    from vena.model.fm.lightning.module import FMLightningModule

    # Stub setup: build a module with minimal config and skip the heavy trunk init.
    cfg = TrunkConfig(checkpoint="/nonexistent.pt", class_token=9)
    monkeypatch.setattr(
        FMLightningModule, "_setup_trunk_and_controlnet", lambda self: None
    )
    monkeypatch.setattr(FMLightningModule, "setup", lambda self, stage=None: None)
    module = FMLightningModule(
        trunk_config=cfg,
        conditioning_specs=["latent:t1pre", "mask:wt:identity"],
        stage="S1",
        loss_cfg={"cfm": {"weight": 1.0}},
    )

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    ckpt: dict = {}
    module.on_save_checkpoint(ckpt)
    assert "rng_state" in ckpt
    assert {"python", "numpy", "torch"} <= set(ckpt["rng_state"].keys())

    # Mutate global RNG, then restore via on_load_checkpoint.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    module.on_load_checkpoint(ckpt)
    # After restore: drawing the same number of samples should match a
    # freshly seeded-from-0 reference.
    p_after = random.random()
    n_after = np.random.rand()
    t_after = torch.rand(1).item()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    # The ckpt was captured *after* seed(0) but before any draws — so after
    # restoring we draw the next number from that state, which equals the
    # first draw from a freshly seeded RNG.
    assert p_after == random.random()
    assert n_after == np.random.rand()
    assert t_after == torch.rand(1).item()

"""Unit tests for the B10 load_warm_start zero-key guard.

Verifies that ``FMLightningModule.load_warm_start`` raises ``RuntimeError``
when the checkpoint contains zero keys that both name-match and shape-match
the destination module — the silent-stock-weight trap (e.g. ``trainable:
false`` warm-started from a ``trainable: true`` checkpoint).

The non-zero-overlap case is covered by ``test_load_warm_start.py``; this
file focuses on the guard path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vena.model.fm.lightning.module import FMLightningModule
from vena.model.fm.maisi.config import TrunkConfig

pytestmark = pytest.mark.unit


def _stub_module(monkeypatch: pytest.MonkeyPatch) -> FMLightningModule:
    monkeypatch.setattr(FMLightningModule, "_setup_trunk_and_controlnet", lambda self: None)
    monkeypatch.setattr(FMLightningModule, "setup", lambda self, stage=None: None)
    return FMLightningModule(
        trunk_config=TrunkConfig(checkpoint="/nonexistent.pt", class_token=9),
        conditioning_specs=["latent:t1pre", "mask:wt:identity"],
        stage="S1",
        loss_cfg={"cfm": {"weight": 1.0}},
    )


def _save_ckpt(path: Path, state_dict: dict) -> None:
    torch.save({"state_dict": state_dict, "epoch": 0, "global_step": 0}, path)


def test_zero_name_overlap_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No matching key name → zero loadable → RuntimeError."""
    module = _stub_module(monkeypatch)
    # Checkpoint uses a key that does not exist in own state_dict.
    ckpt = tmp_path / "trunk_only.ckpt"
    _save_ckpt(ckpt, {"trunk.nonexistent.weight": torch.zeros(4, 4)})

    with pytest.raises(RuntimeError, match="loaded 0 keys"):
        module.load_warm_start(ckpt)


def test_error_message_mentions_trainable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The error message must mention the trainable flag so the user knows what to check."""
    module = _stub_module(monkeypatch)
    ckpt = tmp_path / "no_overlap.ckpt"
    _save_ckpt(ckpt, {"trunk_key_not_in_module": torch.zeros(1)})

    with pytest.raises(RuntimeError, match="trainable"):
        module.load_warm_start(ckpt)

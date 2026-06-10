"""Unit tests for ``FMLightningModule.load_warm_start`` (CPU-only).

Warm-start loads weights only, leaves optimiser / EMA / scheduler / RNG
state untouched. It must:

* successfully transfer keys that overlap between source and destination,
* skip keys present only in the source (logged as ``unexpected``),
* skip keys whose shapes mismatch (logged as ``unexpected``; not loaded),
* leave keys present only in the destination at their initialised values
  (logged as ``missing``),
* raise ``FileNotFoundError`` if the checkpoint path doesn't exist,
* raise ``KeyError`` if the file is not a Lightning checkpoint (no
  ``state_dict`` key).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vena.model.fm.lightning.module import FMLightningModule
from vena.model.fm.maisi.config import TrunkConfig

pytestmark = pytest.mark.unit


def _stub_module(monkeypatch: pytest.MonkeyPatch) -> FMLightningModule:
    """Build a minimal FMLightningModule without touching the real MAISI trunk."""
    monkeypatch.setattr(
        FMLightningModule, "_setup_trunk_and_controlnet", lambda self: None
    )
    monkeypatch.setattr(FMLightningModule, "setup", lambda self, stage=None: None)
    return FMLightningModule(
        trunk_config=TrunkConfig(checkpoint="/nonexistent.pt", class_token=9),
        conditioning_specs=["latent:t1pre", "mask:wt:identity"],
        stage="S1",
        loss_cfg={"cfm": {"weight": 1.0}},
    )


def _save_lightning_like_ckpt(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    torch.save({"state_dict": state_dict, "epoch": 0, "global_step": 0}, path)


def test_load_warm_start_overlapping_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _stub_module(monkeypatch)
    own = module.state_dict()
    # Pick a real key from the destination so the overlap is non-empty.
    target_key = next(iter(own.keys()))
    target_shape = own[target_key].shape

    # Source carries the same key (loadable) + a key the destination lacks
    # (unexpected) + a key with mismatched shape (also unexpected).
    src = {
        target_key: torch.randn(*target_shape),
        "controlnet.does_not_exist.weight": torch.zeros(3),
    }
    # Shape mismatch on a real key only if we can find one (the dummy is enough).
    ckpt_path = tmp_path / "src.ckpt"
    _save_lightning_like_ckpt(ckpt_path, src)

    counts = module.load_warm_start(ckpt_path)
    assert counts["loaded"] == 1
    assert counts["unexpected"] >= 1
    # The destination weight should now equal the source value.
    assert torch.allclose(module.state_dict()[target_key], src[target_key])


def test_load_warm_start_shape_mismatch_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _stub_module(monkeypatch)
    own = module.state_dict()
    target_key = next(iter(own.keys()))
    orig = own[target_key].clone()

    # Pick a deliberately wrong shape so the filter rejects this key.
    wrong_shape = tuple(s + 1 for s in own[target_key].shape)
    src = {target_key: torch.randn(*wrong_shape)}
    ckpt_path = tmp_path / "src.ckpt"
    _save_lightning_like_ckpt(ckpt_path, src)

    counts = module.load_warm_start(ckpt_path)
    assert counts["loaded"] == 0
    assert counts["unexpected"] == 1
    # Destination weight untouched.
    assert torch.allclose(module.state_dict()[target_key], orig)


def test_load_warm_start_missing_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _stub_module(monkeypatch)
    with pytest.raises(FileNotFoundError):
        module.load_warm_start(tmp_path / "nope.ckpt")


def test_load_warm_start_non_lightning_ckpt_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _stub_module(monkeypatch)
    # Plain tensor dict — no ``state_dict`` wrapper.
    p = tmp_path / "raw.pt"
    torch.save({"some": torch.zeros(1)}, p)
    with pytest.raises(KeyError):
        module.load_warm_start(p)

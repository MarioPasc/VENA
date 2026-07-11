"""State-restoration contract for the ``vena_fm`` inference adapter.

Regression cover for the S3 load failure found on 2026-07-11: an S3
(decoder-perceptual-loss) run checkpoints ``lpl_loss.*`` and
``feature_stats.*`` — submodules the *sampling* module never builds, because
:meth:`VenaFMAdapter._build_module` passes no ``lpl_config``. Those keys
therefore surfaced as ``unexpected`` and the adapter raised, so every S3 row in
the benchmark registry died at load. They must be stripped; anything else
unexpected must still raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from vena.inference.adapters.vena_fm_adapter import VenaFMAdapter, VenaFMAdapterError

pytestmark = pytest.mark.unit


class _StubModule:
    """Records what actually reaches ``load_state_dict``."""

    def __init__(self, *, known: set[str]) -> None:
        self.known = known
        self.seen: dict[str, Any] = {}

    def load_state_dict(
        self, state: dict[str, Any], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        self.seen = state
        missing = sorted(self.known - set(state))
        unexpected = sorted(set(state) - self.known)
        return missing, unexpected


def _adapter(checkpoint_path: Path) -> VenaFMAdapter:
    """Build the adapter without running its checkpoint-touching ``__init__``."""
    adapter = object.__new__(VenaFMAdapter)
    adapter.name = "VENA-S3-test"
    adapter.checkpoint_path = checkpoint_path
    adapter.device = torch.device("cpu")
    return adapter


def _write_ckpt(path: Path, state: dict[str, torch.Tensor]) -> None:
    torch.save({"state_dict": state}, path)


# The sampling module owns exactly these prefixes; lpl_loss/feature_stats are
# training-only and are absent by construction.
_SAMPLING_KEYS = {"controlnet.w", "ema.w", "_trunk_module.w", "trunk_ema.w"}


def test_s3_lpl_keys_are_stripped_not_raised(tmp_path: Path) -> None:
    """An S3 checkpoint loads cleanly; the LPL keys never reach the module."""
    ckpt = tmp_path / "ema_best.ckpt"
    _write_ckpt(
        ckpt,
        {
            **{k: torch.zeros(1) for k in _SAMPLING_KEYS},
            "lpl_loss.scale": torch.zeros(1),
            "feature_stats.running_mean": torch.zeros(1),
        },
    )
    module = _StubModule(known=_SAMPLING_KEYS)

    _adapter(ckpt)._load_state_dict(module)

    assert set(module.seen) == _SAMPLING_KEYS
    assert not any(k.startswith(("lpl_loss.", "feature_stats.")) for k in module.seen)


def test_s1_checkpoint_is_unaffected(tmp_path: Path) -> None:
    """An S1 checkpoint carries no LPL state — the strip is a no-op."""
    ckpt = tmp_path / "ema_best.ckpt"
    _write_ckpt(ckpt, {k: torch.zeros(1) for k in _SAMPLING_KEYS})
    module = _StubModule(known=_SAMPLING_KEYS)

    _adapter(ckpt)._load_state_dict(module)

    assert set(module.seen) == _SAMPLING_KEYS


def test_genuinely_unexpected_key_still_raises(tmp_path: Path) -> None:
    """The mismatch guard survives the strip — a wrong architecture is loud."""
    ckpt = tmp_path / "ema_best.ckpt"
    _write_ckpt(
        ckpt,
        {**{k: torch.zeros(1) for k in _SAMPLING_KEYS}, "some_other_head.w": torch.zeros(1)},
    )
    module = _StubModule(known=_SAMPLING_KEYS)

    with pytest.raises(VenaFMAdapterError, match="unexpected keys"):
        _adapter(ckpt)._load_state_dict(module)

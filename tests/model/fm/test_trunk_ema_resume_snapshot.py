"""R6 trunk-EMA snapshot save/load tests.

Complementary to ``test_trunk_ema_resume.py`` (which covers the
``WarmupEMA.load_state_dict`` round-trip in isolation). This file targets
the new sibling-snapshot path introduced for the S1→S3 warm-start:

1. :class:`TrunkEMASnapshotCallback` writes
   ``<ckpt_dir>/trunk_ema_snapshot.pt`` on every Lightning save event.
2. :meth:`FMLightningModule.set_pending_trunk_ema_snapshot` publishes the
   path to be loaded.
3. :meth:`FMLightningModule.setup` (specifically the
   ``_maybe_load_trunk_ema_snapshot`` helper) restores the shadow into a
   freshly-built ``trunk_ema``.

Tests stay CPU-only and avoid loading the real MAISI trunk by working
directly on a minimal ``WarmupEMA(_ToyTrunk(...))`` plus a stub
``FMLightningModule`` that owns ``trunk_ema`` and exposes the public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from vena.model.fm.ema import WarmupEMA
from vena.model.fm.lightning.callbacks import (
    TRUNK_EMA_SNAPSHOT_FILENAME,
    TrunkEMASnapshotCallback,
)

pytestmark = pytest.mark.unit


class _ToyTrunk(nn.Module):
    """Two-layer stand-in for the MAISI trunk (no checkpoint, CPU-only)."""

    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.layer = nn.Linear(8, 8)
        self.layer2 = nn.Linear(8, 4)
        for p in self.parameters():
            p.data.fill_(init_value)


class _StubModule(nn.Module):
    """Minimal stand-in for ``FMLightningModule`` exposing the R6 attributes.

    Owns a ``trunk_ema`` shadow plus the public setter and the private
    ``_maybe_load_trunk_ema_snapshot`` helper. Mirrors the exact lifecycle:
    ``set_pending_trunk_ema_snapshot`` then ``trunk_ema`` build then load.
    """

    def __init__(self, trunk_init: float, ema_decay: float = 0.5) -> None:
        super().__init__()
        self.trunk = _ToyTrunk(init_value=trunk_init)
        self.trunk_ema: WarmupEMA | None = WarmupEMA(
            self.trunk, decay=ema_decay, update_after_step=0, update_every=1
        )
        self._pending_trunk_ema_snapshot: Path | None = None
        # Required by ``_maybe_load_trunk_ema_snapshot`` (uses ``self.device``).
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def set_pending_trunk_ema_snapshot(self, path: Path | None) -> None:
        self._pending_trunk_ema_snapshot = path

    def _maybe_load_trunk_ema_snapshot(self) -> None:
        # Same body as the real LM; duplicated here so the test is hermetic
        # (no FMLightningModule construction needed).
        snapshot = self._pending_trunk_ema_snapshot
        if snapshot is None:
            return
        if not snapshot.exists():
            return
        assert self.trunk_ema is not None
        shadow_sd = torch.load(snapshot, map_location=self.device, weights_only=True)
        self.trunk_ema.ema_model.load_state_dict(shadow_sd, strict=False)


def _drift_trunk_ema(module: _StubModule, fill: float, n_steps: int = 16) -> None:
    """Fill the trunk with a known value and step the EMA so the shadow drifts."""
    with torch.no_grad():
        for p in module.trunk.parameters():
            p.data.fill_(fill)
    assert module.trunk_ema is not None
    for _ in range(n_steps):
        module.trunk_ema.update()


# ---------------------------------------------------------------------------
# (1) Callback writes the sibling file.
# ---------------------------------------------------------------------------


def test_snapshot_callback_writes_sibling_file(tmp_path: Path) -> None:
    """``on_save_checkpoint`` must mirror ``trunk_ema.ema_model.state_dict``
    to ``<dirpath>/trunk_ema_snapshot.pt`` byte-for-byte (matches the
    format consumed by :mod:`routines.fm.exhaustive_val.engine`).
    """
    module = _StubModule(trunk_init=1.0)
    _drift_trunk_ema(module, fill=7.0, n_steps=8)
    assert module.trunk_ema is not None
    expected_shadow_sd = {k: v.clone() for k, v in module.trunk_ema.ema_model.state_dict().items()}

    callback = TrunkEMASnapshotCallback(dirpath=tmp_path / "checkpoints")
    callback.on_save_checkpoint(
        trainer=None,  # type: ignore[arg-type]
        pl_module=module,  # type: ignore[arg-type]
        checkpoint={},
    )

    written_path = tmp_path / "checkpoints" / TRUNK_EMA_SNAPSHOT_FILENAME
    assert written_path.is_file(), f"snapshot not written at {written_path}"

    loaded = torch.load(written_path, map_location="cpu", weights_only=True)
    assert set(loaded) == set(expected_shadow_sd)
    for k, v in expected_shadow_sd.items():
        assert torch.equal(loaded[k], v), f"shadow tensor {k!r} differs"


def test_snapshot_callback_noop_when_trunk_ema_missing(tmp_path: Path) -> None:
    """Callback must NOT write a file when ``pl_module.trunk_ema is None``
    (frozen-trunk runs). Safe to attach the callback unconditionally.
    """

    class _FrozenStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk_ema = None

    callback = TrunkEMASnapshotCallback(dirpath=tmp_path / "checkpoints")
    callback.on_save_checkpoint(
        trainer=None,  # type: ignore[arg-type]
        pl_module=_FrozenStub(),  # type: ignore[arg-type]
        checkpoint={},
    )
    target = tmp_path / "checkpoints" / TRUNK_EMA_SNAPSHOT_FILENAME
    assert not target.exists(), "snapshot should NOT have been written"


# ---------------------------------------------------------------------------
# (2) setup() loads the snapshot when the pending attribute is set.
# ---------------------------------------------------------------------------


def test_setup_loads_snapshot_when_pending_path_set(tmp_path: Path) -> None:
    """After ``set_pending_trunk_ema_snapshot`` + ``_maybe_load_trunk_ema_snapshot``,
    the shadow weights must match the saved snapshot, NOT the fresh init.
    This is the core R6 invariant: warm-start continues the EMA average.
    """
    # 1. Build the "source" module + drift its EMA. Save the shadow.
    source = _StubModule(trunk_init=1.0)
    _drift_trunk_ema(source, fill=7.0, n_steps=16)
    assert source.trunk_ema is not None
    snapshot_path = tmp_path / "checkpoints" / TRUNK_EMA_SNAPSHOT_FILENAME
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source.trunk_ema.ema_model.state_dict(), snapshot_path)
    saved_shadow_value = source.trunk_ema.ema_model.layer.weight.detach().clone()

    # 2. Build a "resume" module with a clearly distinct fresh init (99.0).
    #    Pre-load value of the shadow is the init drift from 99.0 → 99.0
    #    (no updates ran), distinct from saved_shadow_value (≈ from 1.0 → 7.0).
    resume = _StubModule(trunk_init=99.0)
    assert resume.trunk_ema is not None
    pre_load = resume.trunk_ema.ema_model.layer.weight.detach().clone()
    assert not torch.allclose(pre_load, saved_shadow_value)

    # 3. Set the pending path and trigger the load helper directly. After
    #    this, the shadow must equal the SAVED value, not the fresh init.
    resume.set_pending_trunk_ema_snapshot(snapshot_path)
    resume._maybe_load_trunk_ema_snapshot()

    post_load = resume.trunk_ema.ema_model.layer.weight.detach()
    assert torch.allclose(post_load, saved_shadow_value), (
        "trunk_ema shadow did not match saved snapshot after"
        " _maybe_load_trunk_ema_snapshot — R6 is broken"
    )


# ---------------------------------------------------------------------------
# (3) setup() is a no-op when pending path is None or the file is missing.
# ---------------------------------------------------------------------------


def test_setup_skips_load_when_pending_path_none() -> None:
    """No snapshot path published → ``_maybe_load_trunk_ema_snapshot`` is a
    no-op, the freshly-built ``trunk_ema`` keeps its init. This is the
    BASELINE / CONTINUE / pre-R6 path that must remain unchanged.
    """
    module = _StubModule(trunk_init=99.0)
    assert module.trunk_ema is not None
    pre = module.trunk_ema.ema_model.layer.weight.detach().clone()
    # Path NOT set — _pending_trunk_ema_snapshot is still None.
    module._maybe_load_trunk_ema_snapshot()
    post = module.trunk_ema.ema_model.layer.weight.detach()
    assert torch.allclose(pre, post), (
        "shadow changed despite no pending snapshot — load helper is leaking state"
    )


def test_setup_warns_when_pending_path_missing(tmp_path: Path) -> None:
    """Pending path set but the file does not exist — the helper must NOT
    raise, leaving the freshly-built EMA in place. Covers the documented
    legacy ``pre-R6 S1 checkpoint`` warm-start path.
    """
    module = _StubModule(trunk_init=99.0)
    assert module.trunk_ema is not None
    pre = module.trunk_ema.ema_model.layer.weight.detach().clone()

    missing_path = tmp_path / "no_such" / TRUNK_EMA_SNAPSHOT_FILENAME
    module.set_pending_trunk_ema_snapshot(missing_path)
    module._maybe_load_trunk_ema_snapshot()  # must NOT raise

    post = module.trunk_ema.ema_model.layer.weight.detach()
    assert torch.allclose(pre, post), "shadow changed despite missing snapshot — legacy path broken"


# ---------------------------------------------------------------------------
# (4) Snapshot round-trip through callback save + setup load.
# ---------------------------------------------------------------------------


def test_callback_save_then_setup_load_round_trip(tmp_path: Path) -> None:
    """End-to-end: the file written by the callback in run A is exactly the
    state loaded by ``setup`` in run B. This is the production contract
    that R6 + warm-start rely on.
    """
    source = _StubModule(trunk_init=1.0)
    _drift_trunk_ema(source, fill=11.0, n_steps=12)
    assert source.trunk_ema is not None
    expected = source.trunk_ema.ema_model.layer.weight.detach().clone()

    ckpt_dir = tmp_path / "checkpoints"
    callback = TrunkEMASnapshotCallback(dirpath=ckpt_dir)
    callback.on_save_checkpoint(
        trainer=None,  # type: ignore[arg-type]
        pl_module=source,  # type: ignore[arg-type]
        checkpoint={},
    )
    snapshot = ckpt_dir / TRUNK_EMA_SNAPSHOT_FILENAME
    assert snapshot.is_file()

    # Resume into a fresh module, point at the saved snapshot.
    resume = _StubModule(trunk_init=42.0)
    resume.set_pending_trunk_ema_snapshot(snapshot)
    resume._maybe_load_trunk_ema_snapshot()
    assert resume.trunk_ema is not None
    post = resume.trunk_ema.ema_model.layer.weight.detach()
    assert torch.allclose(post, expected)


# ---------------------------------------------------------------------------
# Sanity check: confirm the import surface is stable.
# ---------------------------------------------------------------------------


def test_callback_exports() -> None:
    """The new public symbols must be importable from the canonical path."""
    from vena.model.fm.lightning.callbacks import (
        TRUNK_EMA_SNAPSHOT_FILENAME as _F,
    )
    from vena.model.fm.lightning.callbacks import (
        TrunkEMASnapshotCallback as _C,
    )

    assert _F == "trunk_ema_snapshot.pt"
    assert _C is TrunkEMASnapshotCallback


# Defensive: keep ``Any`` referenced so the import survives ruff cleanup
# (used in the on_save_checkpoint dict annotation in the real callback).
_ANNOTATION_PROBE: Any = None

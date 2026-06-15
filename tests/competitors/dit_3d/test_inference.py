"""Unit tests for the 3D-DiT inference helpers (no GPU required).

Covers:

- ``_resolve_checkpoint`` filename mapping (best / latest / epoch_N).
- ``_resolve_split_patients`` returns ``(pid, pidx)`` pairs for the requested
  split.
- ``_rebuild_dit_from_meta`` produces a model whose state_dict keys match a
  freshly-built sibling at the same architecture.

GPU paths (Euler sample + MAISI decode) are exercised by the per-platform
smoke runs, not here. Marker stays ``unit``.

Citation: arXiv:2212.09748; arXiv:2509.24194.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vena.competitors.dit_3d.inference import (
    InferenceError,
    _resolve_checkpoint,
    _resolve_split_patients,
)

from .test_dataset import _make_synth_latent_h5  # type: ignore[no-redef]

pytestmark = pytest.mark.unit


def _touch_ckpt(run_dir: Path, name: str) -> Path:
    ck = run_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    p = ck / name
    p.write_bytes(b"\x00")  # placeholder; _resolve_checkpoint only checks existence
    return p


def test_resolve_checkpoint_best_and_latest(tmp_path: Path) -> None:
    _touch_ckpt(tmp_path, "best_net_dit.pth")
    _touch_ckpt(tmp_path, "latest_net_dit.pth")
    assert _resolve_checkpoint(tmp_path, "best").name == "best_net_dit.pth"
    assert _resolve_checkpoint(tmp_path, "latest").name == "latest_net_dit.pth"


def test_resolve_checkpoint_epoch_n(tmp_path: Path) -> None:
    _touch_ckpt(tmp_path, "epoch_5_net_dit.pth")
    assert _resolve_checkpoint(tmp_path, 5).name == "epoch_5_net_dit.pth"


def test_resolve_checkpoint_missing_raises(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(InferenceError, match="not found"):
        _resolve_checkpoint(tmp_path, "best")


def test_resolve_split_patients_returns_pid_pidx(tmp_path: Path) -> None:
    h5 = _make_synth_latent_h5(tmp_path / "synth.h5", n=4)
    out = _resolve_split_patients(h5, fold=0, phase="train", n_patients=10)
    assert len(out) == 2
    pids, pidxs = zip(*out)
    assert all(isinstance(p, str) for p in pids)
    assert all(isinstance(i, int) for i in pidxs)


def test_rebuild_dit_from_meta_matches_fresh_build(tmp_path: Path) -> None:
    """``arch_meta`` round-trip: rebuilt model has the same state_dict keys."""
    pytest.importorskip("timm")  # vendored dit3d.py imports timm.{layers,models}
    from vena.competitors.dit_3d.inference import _rebuild_dit_from_meta
    from vena.competitors.dit_3d.runner import _build_dit3d

    meta = {
        "input_size": [8, 8, 8],   # must be divisible by patch_size
        "in_channels": 12,
        "out_channels": 4,
        "hidden_size": 24,        # divisible by 3 and by num_heads
        "depth": 2,
        "num_heads": 6,
        "patch_size": 4,
        "mlp_ratio": 4.0,
    }
    a = _rebuild_dit_from_meta(meta)
    b = _build_dit3d(
        input_size=tuple(meta["input_size"]),
        in_channels=meta["in_channels"],
        out_channels=meta["out_channels"],
        hidden_size=meta["hidden_size"],
        depth=meta["depth"],
        num_heads=meta["num_heads"],
        patch_size=meta["patch_size"],
        mlp_ratio=meta["mlp_ratio"],
    )
    assert sorted(a.state_dict().keys()) == sorted(b.state_dict().keys())

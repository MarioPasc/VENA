"""Unit tests for the 3D-Latent-Pix2Pix inference helpers (no GPU required).

Covers:

- ``_resolve_checkpoint`` filename mapping (best / latest / epoch_N).
- ``_resolve_split_patients`` returns ``(pid, pidx)`` pairs for the requested
  split.
- ``_rebuild_generator_from_meta`` produces a model whose state_dict keys
  match a freshly-built sibling at the same architecture.

GPU paths (G forward + MAISI decode) are exercised by the per-platform smoke
runs, not here. Marker stays ``unit``.

Citation: arXiv:1611.07004; arXiv:2509.24194.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vena.competitors.lpix2pix_3d.inference import (
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
    _touch_ckpt(tmp_path, "best_net_pix2pix.pth")
    _touch_ckpt(tmp_path, "latest_net_pix2pix.pth")
    assert _resolve_checkpoint(tmp_path, "best").name == "best_net_pix2pix.pth"
    assert _resolve_checkpoint(tmp_path, "latest").name == "latest_net_pix2pix.pth"


def test_resolve_checkpoint_epoch_n(tmp_path: Path) -> None:
    _touch_ckpt(tmp_path, "epoch_5_net_pix2pix.pth")
    assert _resolve_checkpoint(tmp_path, 5).name == "epoch_5_net_pix2pix.pth"


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


def test_rebuild_generator_from_meta_matches_fresh_build() -> None:
    """``arch_meta`` round-trip: rebuilt generator has same state_dict keys."""
    pytest.importorskip("monai")
    from vena.competitors.lpix2pix_3d.inference import _rebuild_generator_from_meta
    from vena.competitors.lpix2pix_3d.runner import _build_generator

    meta = {
        "latent_channels": 4,
        "cond_latents": 2,
        "disc_ndf": 64,
        "disc_num_layers": 4,
    }
    a = _rebuild_generator_from_meta(meta)
    b = _build_generator(
        latent_channels=meta["latent_channels"],
        cond_latents=meta["cond_latents"],
    )
    assert sorted(a.state_dict().keys()) == sorted(b.state_dict().keys())


def test_rebuild_discriminator_from_meta_matches_fresh_build() -> None:
    """Discriminator round-trip is similar — required to load D weights at audit."""
    from vena.competitors.lpix2pix_3d.inference import _rebuild_discriminator_from_meta
    from vena.competitors.lpix2pix_3d.runner import _build_discriminator

    meta = {
        "latent_channels": 4,
        "cond_latents": 2,
        "disc_ndf": 64,
        "disc_num_layers": 4,
    }
    a = _rebuild_discriminator_from_meta(meta)
    b = _build_discriminator(
        latent_channels=meta["latent_channels"],
        cond_latents=meta["cond_latents"],
        ndf=meta["disc_ndf"],
        num_layers=meta["disc_num_layers"],
    )
    assert sorted(a.state_dict().keys()) == sorted(b.state_dict().keys())

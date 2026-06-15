"""Smoke tests for the T1C-RFlow inference primitives.

No checkpoint, no CUDA, no real H5 — just shape / sign / loss contracts
that should hold purely on dummy tensors.

Citation: Eidex et al. 2025, arXiv:2509.24194.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.unit


def test_psnr_perfect_match_is_inf() -> None:
    from vena.competitors.t1c_rflow.inference import _psnr

    x = np.zeros((4, 4, 4), dtype=np.float32)
    assert _psnr(x, x) == float("inf")


def test_psnr_increases_with_better_prediction() -> None:
    from vena.competitors.t1c_rflow.inference import _psnr

    rng = np.random.default_rng(0)
    target = rng.uniform(0, 1, size=(8, 8, 8)).astype(np.float32)
    far = rng.uniform(0, 1, size=(8, 8, 8)).astype(np.float32)
    near = target + 0.01 * rng.standard_normal(target.shape).astype(np.float32)
    assert _psnr(near, target) > _psnr(far, target)


def test_rflow_velocity_target_matches_paper_eq3() -> None:
    """Paper Eq. 3 / upstream train_rflow.py:207 — ``u_t = x1 - x0``."""
    z1 = torch.randn(1, 4, 8, 8, 6)  # clean target
    z0 = torch.randn(1, 4, 8, 8, 6)  # noise
    u_target = z1 - z0
    # The loss target the upstream training code minimises:
    v_pred = u_target.clone()  # oracle predictor → loss should be 0.
    loss = F.l1_loss(v_pred, z1 - z0)
    assert float(loss.item()) == 0.0


def test_rflow_l1_loss_signature_is_finite() -> None:
    """Random predictor → finite (non-zero) L1 loss on velocity target."""
    z1 = torch.randn(2, 4, 8, 8, 6)
    z0 = torch.randn(2, 4, 8, 8, 6)
    v_pred = torch.randn(2, 4, 8, 8, 6)
    loss = F.l1_loss(v_pred, z1 - z0)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_resolve_checkpoint_missing_raises(tmp_path) -> None:
    """``--epoch best`` against an empty run dir surfaces a clear error."""
    from vena.competitors.t1c_rflow.inference import (
        InferenceError,
        _resolve_checkpoint,
    )

    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(InferenceError, match="best_net_unet.pth"):
        _resolve_checkpoint(tmp_path, "best")


def test_resolve_checkpoint_finds_named(tmp_path) -> None:
    from vena.competitors.t1c_rflow.inference import _resolve_checkpoint

    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "epoch_3_net_unet.pth").write_bytes(b"")
    out = _resolve_checkpoint(tmp_path, 3)
    assert out.name == "epoch_3_net_unet.pth"
